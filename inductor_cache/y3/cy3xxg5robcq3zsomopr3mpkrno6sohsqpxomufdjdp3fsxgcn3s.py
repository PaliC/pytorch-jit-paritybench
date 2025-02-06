# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/di/cdigxgrsu53y2zae46nxnqpnmokksqck55z2ojcl7upbol4ueiqk.py
# Topologically Sorted Source Nodes: [input_4, batch_norm_1, relu_1], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   input_4 => _low_memory_max_pool2d_with_offsets, getitem_1
#   relu_1 => relu_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %sub_242 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_2402), kwargs = {})
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
    tl.store(out_ptr0 + (x7 + 24576*x3), tmp51, None)
    tl.store(out_ptr1 + (x8), tmp76, None)
    tl.store(out_ptr2 + (x8), tmp93, None)
    tl.store(out_ptr3 + (x8), tmp78, None)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5wne4x4skcrdmnnqv3gz2hsbbdbhks5giwl4apqqk6ysv6lss7.py
# Topologically Sorted Source Nodes: [batch_norm_2, relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_7, mul_8, sub_2
#   relu_2 => relu_2
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
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


# kernel path: inductor_cache/gd/cgdqlpo5tqfg3fr4rmzbnaloxjbcx7u23zib6jpn2jaylzrvydhi.py
# Topologically Sorted Source Nodes: [concated_features_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 24576*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/hz/chzvzwjhr2o2o3loceoqaxdocm6ya2pwt7bbpouxzvv7kvaffsuv.py
# Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_10, mul_11, sub_3
#   relu_3 => relu_3
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
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 96)
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


# kernel path: inductor_cache/5y/c5yxnvwvtasu55bnxoti7m3cpf6gxdlgyjsowhzu4d2x4izf4jw5.py
# Topologically Sorted Source Nodes: [concated_features_2, batch_norm_5, relu_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_5 => add_11, mul_16, mul_17, sub_5
#   concated_features_2 => cat_1
#   relu_5 => relu_5
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2, %convolution_4], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 128)
    x0 = (xindex % 256)
    x2 = xindex // 32768
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 24576*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 8192*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 256*((-96) + x1) + 8192*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/gt/cgtnspgm6kpmjdejvy46aegjccmtcrk52g7a25xokaart44mf4wu.py
# Topologically Sorted Source Nodes: [concated_features_3, batch_norm_7, relu_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_7 => add_15, mul_22, mul_23, sub_7
#   concated_features_3 => cat_2
#   relu_7 => relu_7
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2, %convolution_4, %convolution_6], 1), kwargs = {})
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 160)
    x0 = (xindex % 256)
    x2 = xindex // 40960
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 24576*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 8192*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-96) + x1) + 8192*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 256*((-128) + x1) + 8192*x2), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xtmzbp23xlo2qral3g34lflqmjgdzn3cmptszsgly7nnzwjjm3.py
# Topologically Sorted Source Nodes: [concated_features_4, batch_norm_9, relu_9], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_9 => add_19, mul_28, mul_29, sub_9
#   concated_features_4 => cat_3
#   relu_9 => relu_9
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2, %convolution_4, %convolution_6, %convolution_8], 1), kwargs = {})
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 192)
    x0 = (xindex % 256)
    x2 = xindex // 49152
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 24576*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 8192*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-96) + x1) + 8192*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-128) + x1) + 8192*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 256*((-160) + x1) + 8192*x2), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/5y/c5ykgzvwhovcxbe4ih4a4mndn53seux6z5jissidj4ipnkisdwmg.py
# Topologically Sorted Source Nodes: [concated_features_5, batch_norm_11, relu_11], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_23, mul_34, mul_35, sub_11
#   concated_features_5 => cat_4
#   relu_11 => relu_11
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10], 1), kwargs = {})
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 224)
    x0 = (xindex % 256)
    x2 = xindex // 57344
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 24576*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 8192*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-96) + x1) + 8192*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-128) + x1) + 8192*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 256*((-160) + x1) + 8192*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 224, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 256*((-192) + x1) + 8192*x2), tmp26, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, None)
    tl.store(out_ptr1 + (x3), tmp51, None)
''', device_str='cuda')


# kernel path: inductor_cache/dz/cdzfiascqoqrmdrnoy3px3lxwfevpqj4qwiun3ywt2msafyfabsb.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => cat_5
#   input_6 => add_27, mul_40, mul_41, sub_13
#   input_7 => relu_13
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x2 = xindex // 65536
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 24576*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 8192*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-96) + x1) + 8192*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-128) + x1) + 8192*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 256*((-160) + x1) + 8192*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 224, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 256*((-192) + x1) + 8192*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 256, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 256*((-224) + x1) + 8192*x2), tmp31, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(out_ptr1 + (x3), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/7u/c7uui7rtk6aw2a6jz6p2pqmqgtd5d2uxzfw3bca5hix677chewvf.py
# Topologically Sorted Source Nodes: [input_9, batch_norm_14, relu_14, concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_14 => add_29, mul_43, mul_44, sub_14
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
#   concated_features_16 => cat_15
#   concated_features_17 => cat_16
#   input_10 => cat_17
#   input_9 => avg_pool2d
#   relu_14 => relu_14
# Graph fragment:
#   %avg_pool2d : [num_users=14] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_13, [2, 2], [2, 2]), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33], 1), kwargs = {})
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35], 1), kwargs = {})
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x6 = xindex
    x3 = ((xindex // 64) % 128)
    x4 = xindex // 8192
    x5 = (xindex % 8192)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x6), tmp8, None)
    tl.store(out_ptr1 + (x6), tmp25, None)
    tl.store(out_ptr2 + (x5 + 24576*x4), tmp8, None)
    tl.store(out_ptr3 + (x5 + 26624*x4), tmp8, None)
    tl.store(out_ptr4 + (x5 + 28672*x4), tmp8, None)
    tl.store(out_ptr5 + (x5 + 30720*x4), tmp8, None)
    tl.store(out_ptr6 + (x5 + 32768*x4), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpv27nkm5jkzu77romex4ggfbv37gk6fx4juonn2h4tizaklm5ih.py
# Topologically Sorted Source Nodes: [batch_norm_15, relu_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_15 => add_31, mul_46, mul_47, sub_15
#   relu_15 => relu_15
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
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
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


# kernel path: inductor_cache/p6/cp6jey6lv3qjyfqs55xxozwvi7aybaolutwaykn2howswxckz4va.py
# Topologically Sorted Source Nodes: [concated_features_7, batch_norm_16, relu_16], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_16 => add_33, mul_49, mul_50, sub_16
#   concated_features_7 => cat_6
#   relu_16 => relu_16
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15], 1), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 160)
    x0 = (xindex % 64)
    x2 = xindex // 10240
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp6, other=0.0)
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


# kernel path: inductor_cache/4z/c4zincntexjrqrujmxwrd6zqlldoo4gbg3hwluopjg6xtn3ohblu.py
# Topologically Sorted Source Nodes: [concated_features_8, batch_norm_18, relu_18], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_18 => add_37, mul_55, mul_56, sub_18
#   concated_features_8 => cat_7
#   relu_18 => relu_18
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17], 1), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 192)
    x0 = (xindex % 64)
    x2 = xindex // 12288
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/qo/cqooj6ejb5hcc3ng2z5phq3uey37puh5n7jxxibq46m5t7rhgseo.py
# Topologically Sorted Source Nodes: [concated_features_9, batch_norm_20, relu_20], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_20 => add_41, mul_61, mul_62, sub_20
#   concated_features_9 => cat_8
#   relu_20 => relu_20
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19], 1), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 224)
    x0 = (xindex % 64)
    x2 = xindex // 14336
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 64*((-192) + x1) + 2048*x2), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/ya/cyaczbd4xxs7b4fgd5jc74bmp6acwfvf52pfqzlqnw4gb6w4xsry.py
# Topologically Sorted Source Nodes: [concated_features_10, batch_norm_22, relu_22], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_22 => add_45, mul_67, mul_68, sub_22
#   concated_features_10 => cat_9
#   relu_22 => relu_22
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21], 1), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 256)
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 64*((-192) + x1) + 2048*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 256, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 64*((-224) + x1) + 2048*x2), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgfbhchhknooxt4dvjxoqeegnv6eqyocay4y3pbquoakuhgwjela.py
# Topologically Sorted Source Nodes: [concated_features_11, batch_norm_24, relu_24], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_49, mul_73, mul_74, sub_24
#   concated_features_11 => cat_10
#   relu_24 => relu_24
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23], 1), kwargs = {})
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 288)
    x0 = (xindex % 64)
    x2 = xindex // 18432
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 64*((-192) + x1) + 2048*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 256, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 64*((-224) + x1) + 2048*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 288, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 64*((-256) + x1) + 2048*x2), tmp26, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, None)
    tl.store(out_ptr1 + (x3), tmp51, None)
''', device_str='cuda')


# kernel path: inductor_cache/es/cesdv52w3jvhdoqizzda7n5lfdz5xfnvuc55znh5jcov2dgubkkh.py
# Topologically Sorted Source Nodes: [concated_features_12, batch_norm_26, relu_26], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_26 => add_53, mul_79, mul_80, sub_26
#   concated_features_12 => cat_11
#   relu_26 => relu_26
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25], 1), kwargs = {})
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 320)
    x0 = (xindex % 64)
    x2 = xindex // 20480
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 64*((-192) + x1) + 2048*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 256, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 64*((-224) + x1) + 2048*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 288, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 64*((-256) + x1) + 2048*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 320, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 64*((-288) + x1) + 2048*x2), tmp31, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(out_ptr1 + (x3), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/cp/ccp667pbow56efzhmgdxiwmpoxzoatf4m54sdtkw3yn6kgp57dbt.py
# Topologically Sorted Source Nodes: [concated_features_13, batch_norm_28, relu_28], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_28 => add_57, mul_85, mul_86, sub_28
#   concated_features_13 => cat_12
#   relu_28 => relu_28
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27], 1), kwargs = {})
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 352)
    x0 = (xindex % 64)
    x2 = xindex // 22528
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 64*((-160) + x1) + 2048*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 64*((-192) + x1) + 2048*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 256, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 64*((-224) + x1) + 2048*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 288, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 64*((-256) + x1) + 2048*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 320, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 64*((-288) + x1) + 2048*x2), tmp34, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 352, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 64*((-320) + x1) + 2048*x2), tmp36, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, None)
    tl.store(out_ptr1 + (x3), tmp63, None)
''', device_str='cuda')


# kernel path: inductor_cache/s7/cs7xgrnme3cg46corjxj6vywy22w5x4ghegqyy6exjjfq5ga5kte.py
# Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
#   concated_features_16 => cat_15
#   concated_features_17 => cat_16
#   input_10 => cat_17
# Graph fragment:
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33], 1), kwargs = {})
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35], 1), kwargs = {})
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 24576*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 26624*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 28672*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 30720*x1), tmp0, None)
    tl.store(out_ptr4 + (x0 + 32768*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/ia/cia3sqb72mmwrvxolxhs5yxrc6px6tuvppnixkg4ovck7t2puuge.py
# Topologically Sorted Source Nodes: [batch_norm_30, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_30 => add_61, mul_91, mul_92, sub_30
#   relu_30 => relu_30
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_13, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 384)
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


# kernel path: inductor_cache/kz/ckzxx7rije4fkzy2ujiqgvfzhijlkri3yqfpifvwacipvvtle2eg.py
# Topologically Sorted Source Nodes: [concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_15 => cat_14
#   concated_features_16 => cat_15
#   concated_features_17 => cat_16
#   input_10 => cat_17
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33], 1), kwargs = {})
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35], 1), kwargs = {})
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 26624*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 28672*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 30720*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 32768*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/xr/cxrtd5fwjy3hz2wfzekow4kajkukobe3knphdgbfyhk2q6cg44pf.py
# Topologically Sorted Source Nodes: [batch_norm_32, relu_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_32 => add_65, mul_97, mul_98, sub_32
#   relu_32 => relu_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_14, %unsqueeze_257), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %relu_32 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 416)
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


# kernel path: inductor_cache/gv/cgvvjmafvqzeuwhwzx7c56roaetxn6czdmagg6lrxwc6yp6iyced.py
# Topologically Sorted Source Nodes: [concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_16 => cat_15
#   concated_features_17 => cat_16
#   input_10 => cat_17
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33], 1), kwargs = {})
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35], 1), kwargs = {})
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 28672*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 30720*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 32768*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/b5/cb5rswnmisqjxubwauwcgbnkb4vx3rz6vawvihgrfch6ilxx6fjf.py
# Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_34 => add_69, mul_103, mul_104, sub_34
#   relu_34 => relu_34
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_15, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_34 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 448)
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


# kernel path: inductor_cache/of/cof7yarznk6ty7h2dwibs2lxi7u6q5mixcy5njmje2bs2nfaxhza.py
# Topologically Sorted Source Nodes: [concated_features_17, input_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_17 => cat_16
#   input_10 => cat_17
# Graph fragment:
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35], 1), kwargs = {})
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 30720*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 32768*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5r2z3a4jdql26txiobssktorcqv7wohuzzcbfwtqzvwpdkq6i6o.py
# Topologically Sorted Source Nodes: [batch_norm_36, relu_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_36 => add_73, mul_109, mul_110, sub_36
#   relu_36 => relu_36
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_16, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %relu_36 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_73,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 480)
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


# kernel path: inductor_cache/ya/cyatagwifyncd7kjap42sad53tznsqscatjgv7jft4nw53vbkegu.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_10 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15, %convolution_17, %convolution_19, %convolution_21, %convolution_23, %convolution_25, %convolution_27, %convolution_29, %convolution_31, %convolution_33, %convolution_35, %convolution_37], 1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 32768*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjfsz5dsqc2n2uwotfrqszlabeav6zilj76ifyjyxuqjp6dk6zl.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_11 => add_77, mul_115, mul_116, sub_38
#   input_12 => relu_38
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_17, %unsqueeze_305), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %relu_38 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
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


# kernel path: inductor_cache/5m/c5mdwo3jnqz6uqv3fa2ica5qnfzf4j7knljh6ot23nhawa7bsooq.py
# Topologically Sorted Source Nodes: [input_14, batch_norm_39, relu_39, concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30, concated_features_31, concated_features_32, concated_features_33, concated_features_34, concated_features_35, concated_features_36, concated_features_37, concated_features_38, concated_features_39, concated_features_40, concated_features_41, input_15], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_39 => add_79, mul_118, mul_119, sub_39
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   concated_features_32 => cat_31
#   concated_features_33 => cat_32
#   concated_features_34 => cat_33
#   concated_features_35 => cat_34
#   concated_features_36 => cat_35
#   concated_features_37 => cat_36
#   concated_features_38 => cat_37
#   concated_features_39 => cat_38
#   concated_features_40 => cat_39
#   concated_features_41 => cat_40
#   input_14 => avg_pool2d_1
#   input_15 => cat_41
#   relu_39 => relu_39
# Graph fragment:
#   %avg_pool2d_1 : [num_users=26] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_38, [2, 2], [2, 2]), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_1, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66], 1), kwargs = {})
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68], 1), kwargs = {})
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70], 1), kwargs = {})
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74], 1), kwargs = {})
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76], 1), kwargs = {})
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78], 1), kwargs = {})
#   %cat_38 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80], 1), kwargs = {})
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x6 = xindex
    x3 = ((xindex // 16) % 256)
    x4 = xindex // 4096
    x5 = (xindex % 4096)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x6), tmp8, None)
    tl.store(out_ptr1 + (x6), tmp25, None)
    tl.store(out_ptr2 + (x5 + 8192*x4), tmp8, None)
    tl.store(out_ptr3 + (x5 + 8704*x4), tmp8, None)
    tl.store(out_ptr4 + (x5 + 9216*x4), tmp8, None)
    tl.store(out_ptr5 + (x5 + 9728*x4), tmp8, None)
    tl.store(out_ptr6 + (x5 + 10240*x4), tmp8, None)
    tl.store(out_ptr7 + (x5 + 10752*x4), tmp8, None)
    tl.store(out_ptr8 + (x5 + 11264*x4), tmp8, None)
    tl.store(out_ptr9 + (x5 + 11776*x4), tmp8, None)
    tl.store(out_ptr10 + (x5 + 12288*x4), tmp8, None)
    tl.store(out_ptr11 + (x5 + 12800*x4), tmp8, None)
    tl.store(out_ptr12 + (x5 + 13312*x4), tmp8, None)
    tl.store(out_ptr13 + (x5 + 13824*x4), tmp8, None)
    tl.store(out_ptr14 + (x5 + 14336*x4), tmp8, None)
    tl.store(out_ptr15 + (x5 + 14848*x4), tmp8, None)
    tl.store(out_ptr16 + (x5 + 15360*x4), tmp8, None)
    tl.store(out_ptr17 + (x5 + 15872*x4), tmp8, None)
    tl.store(out_ptr18 + (x5 + 16384*x4), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/bi/cbigz44apib2ww6vvtwojvsxmefozcbtm6ahf64ppsh3uqrvj4dr.py
# Topologically Sorted Source Nodes: [batch_norm_40, relu_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_40 => add_81, mul_121, mul_122, sub_40
#   relu_40 => relu_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_321), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %unsqueeze_325), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %unsqueeze_327), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_81,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
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


# kernel path: inductor_cache/6r/c6r4winp4uamixapctlnskkbh6hwxmprsuottqo2azd6yuq6yhnf.py
# Topologically Sorted Source Nodes: [concated_features_19, batch_norm_41, relu_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_41 => add_83, mul_124, mul_125, sub_41
#   concated_features_19 => cat_18
#   relu_41 => relu_41
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40], 1), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_18, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 288)
    x0 = (xindex % 16)
    x2 = xindex // 4608
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp6 & xmask, other=0.0)
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


# kernel path: inductor_cache/yh/cyhaj2tvpyhqmvd24jzctt7vsi7uwonb4rrbjnyiojzys3acv6zn.py
# Topologically Sorted Source Nodes: [concated_features_20, batch_norm_43, relu_43], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_43 => add_87, mul_130, mul_131, sub_43
#   concated_features_20 => cat_19
#   relu_43 => relu_43
# Graph fragment:
#   %cat_19 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42], 1), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 320)
    x0 = (xindex % 16)
    x2 = xindex // 5120
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4hlpfhxskcoepivpwoqojmx56lks2ale7f2vklhei375fqslnxo.py
# Topologically Sorted Source Nodes: [concated_features_21, batch_norm_45, relu_45], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_45 => add_91, mul_136, mul_137, sub_45
#   concated_features_21 => cat_20
#   relu_45 => relu_45
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44], 1), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_20, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 352)
    x0 = (xindex % 16)
    x2 = xindex // 5632
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 352, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 16*((-320) + x1) + 512*x2), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrnqqc43pudfoe7rocih5ynzopo53yk5udhh5rhbsd6zmzaoaes.py
# Topologically Sorted Source Nodes: [concated_features_22, batch_norm_47, relu_47], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_47 => add_95, mul_142, mul_143, sub_47
#   concated_features_22 => cat_21
#   relu_47 => relu_47
# Graph fragment:
#   %cat_21 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46], 1), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %unsqueeze_377), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %unsqueeze_381), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %unsqueeze_383), kwargs = {})
#   %relu_47 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 384)
    x0 = (xindex % 16)
    x2 = xindex // 6144
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 352, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 16*((-320) + x1) + 512*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 16*((-352) + x1) + 512*x2), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/ug/cugnjqjhfm5v4kwxhiqcjartc2v2me3navvdkllccdmyvzi5ejeg.py
# Topologically Sorted Source Nodes: [concated_features_23, batch_norm_49, relu_49], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_49 => add_99, mul_148, mul_149, sub_49
#   concated_features_23 => cat_22
#   relu_49 => relu_49
# Graph fragment:
#   %cat_22 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48], 1), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_22, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %relu_49 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_99,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 416)
    x0 = (xindex % 16)
    x2 = xindex // 6656
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 352, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 16*((-320) + x1) + 512*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 16*((-352) + x1) + 512*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 416, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 16*((-384) + x1) + 512*x2), tmp26 & xmask, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
    tl.store(out_ptr1 + (x3), tmp51, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zp/czpxdto7yqqzuyhmgpbszpomfkxjdmtjdt2erro2yorokxtfuama.py
# Topologically Sorted Source Nodes: [concated_features_24, batch_norm_51, relu_51], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_51 => add_103, mul_154, mul_155, sub_51
#   concated_features_24 => cat_23
#   relu_51 => relu_51
# Graph fragment:
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50], 1), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_23, %unsqueeze_409), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_413), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_415), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_103,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 448)
    x0 = (xindex % 16)
    x2 = xindex // 7168
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 352, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 16*((-320) + x1) + 512*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 16*((-352) + x1) + 512*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 416, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 16*((-384) + x1) + 512*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 448, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 16*((-416) + x1) + 512*x2), tmp31, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(out_ptr1 + (x3), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/bb/cbbz6sbzsn2bszowfzea7y4mou6ffyk2t2hr6hyx54cbjkrocmal.py
# Topologically Sorted Source Nodes: [concated_features_25, batch_norm_53, relu_53], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_53 => add_107, mul_160, mul_161, sub_53
#   concated_features_25 => cat_24
#   relu_53 => relu_53
# Graph fragment:
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52], 1), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_24, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 480)
    x0 = (xindex % 16)
    x2 = xindex // 7680
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-288) + x1) + 512*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 352, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 16*((-320) + x1) + 512*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 16*((-352) + x1) + 512*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 416, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 16*((-384) + x1) + 512*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 448, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 16*((-416) + x1) + 512*x2), tmp34 & xmask, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 480, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 16*((-448) + x1) + 512*x2), tmp36 & xmask, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, xmask)
    tl.store(out_ptr1 + (x3), tmp63, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n2/cn2cehwr4igkauhp4cuuasinrpldgfpncj6shu4dpk4tevubqcxk.py
# Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
# Graph fragment:
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 8704*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 9216*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 9728*x1), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + 10240*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4qrjhjydufq75lj24q2edmvca6njibdgakb5ledsun4k65y3di.py
# Topologically Sorted Source Nodes: [batch_norm_55, relu_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_55 => add_111, mul_166, mul_167, sub_55
#   relu_55 => relu_55
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_25, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
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


# kernel path: inductor_cache/et/cetombej6d5b2yvhjgsxztgyxm3hvcg3su3apbiaqqkly4ciytnv.py
# Topologically Sorted Source Nodes: [concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 8704*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 9216*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 9728*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 10240*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sr/csr3cqh5nnmegm6d4fcu2kmsstcdqlrrei64s2pwrydilackj52v.py
# Topologically Sorted Source Nodes: [batch_norm_57, relu_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_57 => add_115, mul_172, mul_173, sub_57
#   relu_57 => relu_57
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 544)
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


# kernel path: inductor_cache/x4/cx4chriy6ifmh7lf2mu67i6x2dgs2cnslf6rs2dp63vniuwhjkwy.py
# Topologically Sorted Source Nodes: [concated_features_28, concated_features_29, concated_features_30, concated_features_31], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
# Graph fragment:
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64], 1), kwargs = {})
triton_poi_fused_cat_42 = async_compile.triton('triton_poi_fused_cat_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_42(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 9216*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 9728*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 10240*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 10752*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zu/czugtdtafpfkmhfdcieaoboos5fzae2hkcr7jk6l5n5agiwtb2m7.py
# Topologically Sorted Source Nodes: [batch_norm_59, relu_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_59 => add_119, mul_178, mul_179, sub_59
#   relu_59 => relu_59
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %unsqueeze_473), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_477), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_479), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_119,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 576)
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


# kernel path: inductor_cache/yg/cygypprmfk47pp76ns44vk67efsw67ks2ogzvbxb2pynhavwdptt.py
# Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, concated_features_32], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   concated_features_32 => cat_31
# Graph fragment:
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66], 1), kwargs = {})
triton_poi_fused_cat_44 = async_compile.triton('triton_poi_fused_cat_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_44(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 9728*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 10240*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 10752*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 11264*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ud/cudd4hcxx2dyhzovvtdrpbcf5iazxgbv7jmmlxqzzs7kvtcgw6uu.py
# Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_61 => add_123, mul_184, mul_185, sub_61
#   relu_61 => relu_61
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_28, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 608)
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


# kernel path: inductor_cache/a2/ca22e26d55jwdkygxgeg3obhkdrcy4z6y6gofny7t63sudg5mv3f.py
# Topologically Sorted Source Nodes: [concated_features_30, concated_features_31, concated_features_32, concated_features_33], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   concated_features_32 => cat_31
#   concated_features_33 => cat_32
# Graph fragment:
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66], 1), kwargs = {})
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68], 1), kwargs = {})
triton_poi_fused_cat_46 = async_compile.triton('triton_poi_fused_cat_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_46(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 10240*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 10752*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 11264*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 11776*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvphnb5fkwlqxoo3gplk6gl362r7a4wipkac56gt6bdn2j77uten.py
# Topologically Sorted Source Nodes: [batch_norm_63, relu_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_63 => add_127, mul_190, mul_191, sub_63
#   relu_63 => relu_63
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_29, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_127,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 640)
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


# kernel path: inductor_cache/e3/ce3qxjkuwepnbozc2u3ma5k6cowbwhflfz4fp4mokvwwbstposol.py
# Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_31 => cat_30
#   concated_features_32 => cat_31
#   concated_features_33 => cat_32
#   concated_features_34 => cat_33
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66], 1), kwargs = {})
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68], 1), kwargs = {})
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 10752*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 11264*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 11776*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 12288*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rk/crk34dv47eidofqtoxqnx6tvfnyk3mmkz637qavh4hzakqixzm2o.py
# Topologically Sorted Source Nodes: [batch_norm_65, relu_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_65 => add_131, mul_196, mul_197, sub_65
#   relu_65 => relu_65
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_30, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_65 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
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
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 672)
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


# kernel path: inductor_cache/e6/ce6w3l6qvmnf5z66v6uxw27lszlenvgintamkbcyyyf7w5ymnshs.py
# Topologically Sorted Source Nodes: [concated_features_32, concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_32 => cat_31
#   concated_features_33 => cat_32
#   concated_features_34 => cat_33
#   concated_features_35 => cat_34
# Graph fragment:
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66], 1), kwargs = {})
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68], 1), kwargs = {})
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70], 1), kwargs = {})
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
triton_poi_fused_cat_50 = async_compile.triton('triton_poi_fused_cat_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_50(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 11264*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 11776*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 12288*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 12800*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mk/cmkrsiyfzhbqkmogqrony5cns2yjz3z5q5vd5yaz2ihacicxz27i.py
# Topologically Sorted Source Nodes: [batch_norm_67, relu_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_67 => add_135, mul_202, mul_203, sub_67
#   relu_67 => relu_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_31, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_67 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 704)
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


# kernel path: inductor_cache/2f/c2f6vsmox5igvps4gtcj6tzwe3bkxsrx3rg2yt7urer6qwabl32u.py
# Topologically Sorted Source Nodes: [concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_33 => cat_32
#   concated_features_34 => cat_33
#   concated_features_35 => cat_34
# Graph fragment:
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68], 1), kwargs = {})
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70], 1), kwargs = {})
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
triton_poi_fused_cat_52 = async_compile.triton('triton_poi_fused_cat_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_52(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 11776*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 12288*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 12800*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgq2h52virtfy5qihpcgco3q3snp5pvik6eipwfaewsy7qede4k.py
# Topologically Sorted Source Nodes: [batch_norm_69, relu_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_69 => add_139, mul_208, mul_209, sub_69
#   relu_69 => relu_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_32, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %relu_69 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_139,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 736)
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


# kernel path: inductor_cache/e6/ce64pceetdcw3cxyqm2dx5a5jbtu2pqhwv6s6g3nuo6c6gars2wr.py
# Topologically Sorted Source Nodes: [concated_features_34, concated_features_35, concated_features_36], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_34 => cat_33
#   concated_features_35 => cat_34
#   concated_features_36 => cat_35
# Graph fragment:
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70], 1), kwargs = {})
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 12288*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 12800*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 13312*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvkdkyedbthe3vii4qfy5tm2pojfd7hw5ttrsvsjq7swmvfetov.py
# Topologically Sorted Source Nodes: [batch_norm_71, relu_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_71 => add_143, mul_214, mul_215, sub_71
#   relu_71 => relu_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_33, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
#   %relu_71 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 768)
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


# kernel path: inductor_cache/sa/csa74oifqzrknd2jcnfifrmhvrup3pmw2kvu2rkwiy5b3olql4nc.py
# Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_35 => cat_34
#   concated_features_36 => cat_35
#   concated_features_37 => cat_36
# Graph fragment:
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74], 1), kwargs = {})
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76], 1), kwargs = {})
triton_poi_fused_cat_56 = async_compile.triton('triton_poi_fused_cat_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_56(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 12800*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 13312*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 13824*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c7/cc7hqmexrda5p5mnfgzyl57hn26ymmj3yyxnexiqayzu6shnvukr.py
# Topologically Sorted Source Nodes: [batch_norm_73, relu_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_73 => add_147, mul_220, mul_221, sub_73
#   relu_73 => relu_73
# Graph fragment:
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_34, %unsqueeze_585), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %unsqueeze_589), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %unsqueeze_591), kwargs = {})
#   %relu_73 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 800)
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


# kernel path: inductor_cache/ue/cue3hhgkhrvudicaujsqodegdkl7lh67l6y5owa37jxyvyik4jrg.py
# Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_36 => cat_35
#   concated_features_37 => cat_36
#   concated_features_38 => cat_37
# Graph fragment:
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74], 1), kwargs = {})
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76], 1), kwargs = {})
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78], 1), kwargs = {})
triton_poi_fused_cat_58 = async_compile.triton('triton_poi_fused_cat_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_58(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 13312*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 13824*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 14336*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sa/csalqfjki34ec6velc5qcfwuakefnlbaxkln7pfufijpjkaoyqtv.py
# Topologically Sorted Source Nodes: [batch_norm_75, relu_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_75 => add_151, mul_226, mul_227, sub_75
#   relu_75 => relu_75
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_35, %unsqueeze_601), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_605), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_607), kwargs = {})
#   %relu_75 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 832)
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


# kernel path: inductor_cache/gd/cgdqppeapey4biddmswbu35iuz7stynjdsmnic7kxfvrf4v6sd3m.py
# Topologically Sorted Source Nodes: [concated_features_37, concated_features_38, concated_features_39], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_37 => cat_36
#   concated_features_38 => cat_37
#   concated_features_39 => cat_38
# Graph fragment:
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76], 1), kwargs = {})
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78], 1), kwargs = {})
#   %cat_38 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80], 1), kwargs = {})
triton_poi_fused_cat_60 = async_compile.triton('triton_poi_fused_cat_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_60(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 13824*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 14336*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 14848*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6y/c6yr4tylhvwb5oc7etjoyiytp6wdbsj5madlzcf2tsrwwdvopnph.py
# Topologically Sorted Source Nodes: [batch_norm_77, relu_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_77 => add_155, mul_232, mul_233, sub_77
#   relu_77 => relu_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_36, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 864)
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


# kernel path: inductor_cache/tf/ctfvl2ff2acqtrmkvtwvf6647ancas7nghyeyxo3zl7kzuwynzsb.py
# Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_38 => cat_37
#   concated_features_39 => cat_38
#   concated_features_40 => cat_39
# Graph fragment:
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78], 1), kwargs = {})
#   %cat_38 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80], 1), kwargs = {})
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
triton_poi_fused_cat_62 = async_compile.triton('triton_poi_fused_cat_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_62(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 14336*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 14848*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 15360*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ce/cceapbzgf4tgu5i457e75dpp7sy7i3uqcifacvccgpqqsnbkaxlw.py
# Topologically Sorted Source Nodes: [batch_norm_79, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_79 => add_159, mul_238, mul_239, sub_79
#   relu_79 => relu_79
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_37, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_79 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 896)
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


# kernel path: inductor_cache/w3/cw3zwtgpbu5lznik63ssoibhbwjhczh4y6yxg4dox55aa5knk26v.py
# Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_39 => cat_38
#   concated_features_40 => cat_39
#   concated_features_41 => cat_40
# Graph fragment:
#   %cat_38 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80], 1), kwargs = {})
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
triton_poi_fused_cat_64 = async_compile.triton('triton_poi_fused_cat_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_64(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 14848*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 15360*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 15872*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3i/c3ibuadq7n2ck52ugk5ailnc5hckzb7r4uirwfww34t35ahdgabc.py
# Topologically Sorted Source Nodes: [batch_norm_81, relu_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_81 => add_163, mul_244, mul_245, sub_81
#   relu_81 => relu_81
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_38, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %relu_81 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_65', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 928)
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


# kernel path: inductor_cache/7q/c7qkzclamh2klwdl6rgje4nspkc6exibz7bhxvpijf73jbjrhzio.py
# Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, input_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_40 => cat_39
#   concated_features_41 => cat_40
#   input_15 => cat_41
# Graph fragment:
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
triton_poi_fused_cat_66 = async_compile.triton('triton_poi_fused_cat_66', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_66(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 15360*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 15872*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 16384*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xs/cxsv2bj2yyha6t5c5w5bjaegtb2brlwdw2rhgqqdeu7lmb6b6ha2.py
# Topologically Sorted Source Nodes: [batch_norm_83, relu_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_83 => add_167, mul_250, mul_251, sub_83
#   relu_83 => relu_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_39, %unsqueeze_665), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_669), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_671), kwargs = {})
#   %relu_83 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 960)
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


# kernel path: inductor_cache/7h/c7hram5p2hhhc7ci7pdojnyifi4gpnewdjvc7jkjmdduvdvrhbmo.py
# Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_41 => cat_40
#   input_15 => cat_41
# Graph fragment:
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
triton_poi_fused_cat_68 = async_compile.triton('triton_poi_fused_cat_68', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_68', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_68(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 15872*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 16384*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lq/clqd3g7wbh3cedaklgfwj35ttqnje6buv2iyi54qt3pkbhk67zrd.py
# Topologically Sorted Source Nodes: [batch_norm_85, relu_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_85 => add_171, mul_256, mul_257, sub_85
#   relu_85 => relu_85
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_40, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_85 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_171,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_69 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_69', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 992)
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


# kernel path: inductor_cache/52/c5274sci6tj33knxyijunk6nhd67zqe43e4ynzojc6h6xkm4sjgx.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_15 => cat_41
# Graph fragment:
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_40, %convolution_42, %convolution_44, %convolution_46, %convolution_48, %convolution_50, %convolution_52, %convolution_54, %convolution_56, %convolution_58, %convolution_60, %convolution_62, %convolution_64, %convolution_66, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
triton_poi_fused_cat_70 = async_compile.triton('triton_poi_fused_cat_70', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_70(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 16384*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chgp4aipbtqune3722kr3vutblalnt2s7silptv4mmjep67bnqjq.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_175, mul_262, mul_263, sub_87
#   input_17 => relu_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_41, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_175,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_71 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_71', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 1024)
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


# kernel path: inductor_cache/ch/cch6dsmgz4jhnzsjwkss36aswq22nceij5qpzm2snsvb3s2tz7hx.py
# Topologically Sorted Source Nodes: [input_19, batch_norm_88, relu_88, concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54, concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_88 => add_177, mul_265, mul_266, sub_88
#   concated_features_50 => cat_49
#   concated_features_51 => cat_50
#   concated_features_52 => cat_51
#   concated_features_53 => cat_52
#   concated_features_54 => cat_53
#   concated_features_55 => cat_54
#   concated_features_56 => cat_55
#   concated_features_57 => cat_56
#   input_19 => avg_pool2d_2
#   input_20 => cat_57
#   relu_88 => relu_88
# Graph fragment:
#   %avg_pool2d_2 : [num_users=18] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_87, [2, 2], [2, 2]), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_2, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %relu_88 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
#   %cat_49 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103], 1), kwargs = {})
#   %cat_50 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105], 1), kwargs = {})
#   %cat_51 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107], 1), kwargs = {})
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109], 1), kwargs = {})
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
#   %cat_54 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113], 1), kwargs = {})
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115], 1), kwargs = {})
#   %cat_56 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117], 1), kwargs = {})
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117, %convolution_119], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_72 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_72', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = xindex // 2
    x6 = xindex
    x3 = ((xindex // 4) % 512)
    x4 = xindex // 2048
    x5 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x6), tmp8, None)
    tl.store(out_ptr1 + (x6), tmp25, None)
    tl.store(out_ptr2 + (x5 + 3072*x4), tmp8, None)
    tl.store(out_ptr3 + (x5 + 3200*x4), tmp8, None)
    tl.store(out_ptr4 + (x5 + 3328*x4), tmp8, None)
    tl.store(out_ptr5 + (x5 + 3456*x4), tmp8, None)
    tl.store(out_ptr6 + (x5 + 3584*x4), tmp8, None)
    tl.store(out_ptr7 + (x5 + 3712*x4), tmp8, None)
    tl.store(out_ptr8 + (x5 + 3840*x4), tmp8, None)
    tl.store(out_ptr9 + (x5 + 3968*x4), tmp8, None)
    tl.store(out_ptr10 + (x5 + 4096*x4), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgfhlaqsk222agc766qzgzrylurinhaodlwfih2727ioushda3f.py
# Topologically Sorted Source Nodes: [batch_norm_89, relu_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_89 => add_179, mul_268, mul_269, sub_89
#   relu_89 => relu_89
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_713), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_717), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_719), kwargs = {})
#   %relu_89 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_179,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_73 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_73', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_73(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 128)
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


# kernel path: inductor_cache/fy/cfys3c7d2re2jxqfsdugu2zho4bgjxj2u73c24ndv7wa4pzmrg5p.py
# Topologically Sorted Source Nodes: [concated_features_43, batch_norm_90, relu_90], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_90 => add_181, mul_271, mul_272, sub_90
#   concated_features_43 => cat_42
#   relu_90 => relu_90
# Graph fragment:
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89], 1), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_42, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %relu_90 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_181,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_74 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_74', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 544)
    x0 = (xindex % 4)
    x2 = xindex // 2176
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp6 & xmask, other=0.0)
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


# kernel path: inductor_cache/2k/c2kvulgjdaskv62udp2s35irjbftbno7ukeubs3qqwqjhhiga2nv.py
# Topologically Sorted Source Nodes: [concated_features_44, batch_norm_92, relu_92], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_92 => add_185, mul_277, mul_278, sub_92
#   concated_features_44 => cat_43
#   relu_92 => relu_92
# Graph fragment:
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91], 1), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_43, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
#   %relu_92 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_185,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_75 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_75', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_75', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_75(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 576)
    x0 = (xindex % 4)
    x2 = xindex // 2304
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp11 & xmask, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrod4yxdb2taz6lfr2oeruz63hz5lkqend52b7fe47od6a75axu.py
# Topologically Sorted Source Nodes: [concated_features_45, batch_norm_94, relu_94], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_94 => add_189, mul_283, mul_284, sub_94
#   concated_features_45 => cat_44
#   relu_94 => relu_94
# Graph fragment:
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93], 1), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_44, %unsqueeze_753), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %relu_94 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_189,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 608)
    x0 = (xindex % 4)
    x2 = xindex // 2432
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jw/cjwlotuzpzpbu3zehtj2eawt4nharqxukywj54fcw7aixr4cktbz.py
# Topologically Sorted Source Nodes: [concated_features_46, batch_norm_96, relu_96], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_96 => add_193, mul_289, mul_290, sub_96
#   concated_features_46 => cat_45
#   relu_96 => relu_96
# Graph fragment:
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95], 1), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_45, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %relu_96 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_77 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_77', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_77', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_77(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 640)
    x0 = (xindex % 4)
    x2 = xindex // 2560
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 640, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 4*((-608) + x1) + 128*x2), tmp21 & xmask, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7w/c7wquddpscxbsfwio6uzasfqiumo2nuymop4wivmfmoyg4erwnik.py
# Topologically Sorted Source Nodes: [concated_features_47, batch_norm_98, relu_98], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_98 => add_197, mul_295, mul_296, sub_98
#   concated_features_47 => cat_46
#   relu_98 => relu_98
# Graph fragment:
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97], 1), kwargs = {})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_46, %unsqueeze_785), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_295, %unsqueeze_789), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %unsqueeze_791), kwargs = {})
#   %relu_98 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_197,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_78', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_78', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 672)
    x0 = (xindex % 4)
    x2 = xindex // 2688
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 640, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4*((-608) + x1) + 128*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 672, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 4*((-640) + x1) + 128*x2), tmp26 & xmask, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
    tl.store(out_ptr1 + (x3), tmp51, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2f/c2filii3shcv6havbe5hdigz2oxuirwxaix6sdk626bn6m56tkep.py
# Topologically Sorted Source Nodes: [concated_features_48, batch_norm_100, relu_100], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_100 => add_201, mul_301, mul_302, sub_100
#   concated_features_48 => cat_47
#   relu_100 => relu_100
# Graph fragment:
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99], 1), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_47, %unsqueeze_801), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_805), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_807), kwargs = {})
#   %relu_100 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_201,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 704)
    x0 = (xindex % 4)
    x2 = xindex // 2816
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 640, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4*((-608) + x1) + 128*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 672, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 4*((-640) + x1) + 128*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 704, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 4*((-672) + x1) + 128*x2), tmp31 & xmask, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, xmask)
    tl.store(out_ptr1 + (x3), tmp57, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ar/carxrvhbfwxnprgqfrcjch6qrpxjjtsarwjdw7dks6ehzswhyyat.py
# Topologically Sorted Source Nodes: [concated_features_49, batch_norm_102, relu_102], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_102 => add_205, mul_307, mul_308, sub_102
#   concated_features_49 => cat_48
#   relu_102 => relu_102
# Graph fragment:
#   %cat_48 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101], 1), kwargs = {})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_48, %unsqueeze_817), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
#   %relu_102 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_205,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_80', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_80', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_80(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 736)
    x0 = (xindex % 4)
    x2 = xindex // 2944
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 640, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4*((-608) + x1) + 128*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 672, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 4*((-640) + x1) + 128*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 704, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 4*((-672) + x1) + 128*x2), tmp34 & xmask, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 736, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 4*((-704) + x1) + 128*x2), tmp36 & xmask, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, xmask)
    tl.store(out_ptr1 + (x3), tmp63, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjqnf2lybo6ypics36v426piiwttwv4mneqyqqpz6upoejcd4rvf.py
# Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_50 => cat_49
#   concated_features_51 => cat_50
#   concated_features_52 => cat_51
#   concated_features_53 => cat_52
#   concated_features_54 => cat_53
# Graph fragment:
#   %cat_49 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103], 1), kwargs = {})
#   %cat_50 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105], 1), kwargs = {})
#   %cat_51 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107], 1), kwargs = {})
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109], 1), kwargs = {})
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
triton_poi_fused_cat_81 = async_compile.triton('triton_poi_fused_cat_81', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_81(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3072*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3200*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3328*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 3456*x1), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + 3584*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w6/cw6docbzbpq3rmigjmbmykdkowd7bfgrokxcgllldp2idg2tpgus.py
# Topologically Sorted Source Nodes: [batch_norm_104, relu_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_104 => add_209, mul_313, mul_314, sub_104
#   relu_104 => relu_104
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_49, %unsqueeze_833), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_837), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_839), kwargs = {})
#   %relu_104 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_209,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_82 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_82', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_82', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 768)
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


# kernel path: inductor_cache/t5/ct5bnpg2dr4k34xyia3fcfprftcpxjinbkbmegaitthrncaprbgf.py
# Topologically Sorted Source Nodes: [concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_51 => cat_50
#   concated_features_52 => cat_51
#   concated_features_53 => cat_52
#   concated_features_54 => cat_53
# Graph fragment:
#   %cat_50 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105], 1), kwargs = {})
#   %cat_51 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107], 1), kwargs = {})
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109], 1), kwargs = {})
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
triton_poi_fused_cat_83 = async_compile.triton('triton_poi_fused_cat_83', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_83', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_83(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3200*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3328*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3456*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 3584*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck52r23x75wggtyguom3y7m72qk2jqxw5sgswyn4kavzbtmzbkga.py
# Topologically Sorted Source Nodes: [batch_norm_106, relu_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_106 => add_213, mul_319, mul_320, sub_106
#   relu_106 => relu_106
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_50, %unsqueeze_849), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %unsqueeze_853), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_320, %unsqueeze_855), kwargs = {})
#   %relu_106 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_213,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_84', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_84', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_84(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 800)
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


# kernel path: inductor_cache/l6/cl6kqavj6xmxd5ippvqenx2ocimgffcbfv6o6bfl3kkw4tafyaog.py
# Topologically Sorted Source Nodes: [concated_features_52, concated_features_53, concated_features_54, concated_features_55], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_52 => cat_51
#   concated_features_53 => cat_52
#   concated_features_54 => cat_53
#   concated_features_55 => cat_54
# Graph fragment:
#   %cat_51 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107], 1), kwargs = {})
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109], 1), kwargs = {})
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
#   %cat_54 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113], 1), kwargs = {})
triton_poi_fused_cat_85 = async_compile.triton('triton_poi_fused_cat_85', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_85', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_85(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3328*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3456*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3584*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 3712*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ua/cuatfucx7pehchlon644hzteobp3tvfrxwg2zsfsfloj6o6bxtjx.py
# Topologically Sorted Source Nodes: [batch_norm_108, relu_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_108 => add_217, mul_325, mul_326, sub_108
#   relu_108 => relu_108
# Graph fragment:
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_51, %unsqueeze_865), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_869), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_871), kwargs = {})
#   %relu_108 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_217,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_86 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_86', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_86', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 832)
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


# kernel path: inductor_cache/tb/ctb6bm7x2cpz53ht5r42jj2yrt653x334mhoahraq2xyu322xt33.py
# Topologically Sorted Source Nodes: [concated_features_53, concated_features_54, concated_features_55, concated_features_56], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_53 => cat_52
#   concated_features_54 => cat_53
#   concated_features_55 => cat_54
#   concated_features_56 => cat_55
# Graph fragment:
#   %cat_52 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109], 1), kwargs = {})
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
#   %cat_54 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113], 1), kwargs = {})
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115], 1), kwargs = {})
triton_poi_fused_cat_87 = async_compile.triton('triton_poi_fused_cat_87', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_87', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_87(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3456*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3584*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3712*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 3840*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjkijqc4drok7lalseaijwsswgdcmy4lwv4pc623pl4rrwvbbc2a.py
# Topologically Sorted Source Nodes: [batch_norm_110, relu_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_110 => add_221, mul_331, mul_332, sub_110
#   relu_110 => relu_110
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_52, %unsqueeze_881), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_883), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_885), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_887), kwargs = {})
#   %relu_110 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_88 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_88', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_88(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 864)
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


# kernel path: inductor_cache/72/c72ydw6dgl66a3ybylgc3tj2rgm3uii5xfhlbsnwo2qogajqguxf.py
# Topologically Sorted Source Nodes: [concated_features_54, concated_features_55, concated_features_56, concated_features_57], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_54 => cat_53
#   concated_features_55 => cat_54
#   concated_features_56 => cat_55
#   concated_features_57 => cat_56
# Graph fragment:
#   %cat_53 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111], 1), kwargs = {})
#   %cat_54 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113], 1), kwargs = {})
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115], 1), kwargs = {})
#   %cat_56 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117], 1), kwargs = {})
triton_poi_fused_cat_89 = async_compile.triton('triton_poi_fused_cat_89', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_89', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_89(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3584*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3712*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3840*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 3968*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwx3gfua2gtv62z5yrxoatzxjv3hiopinkg5cvi5gry37bwqvht.py
# Topologically Sorted Source Nodes: [batch_norm_112, relu_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_112 => add_225, mul_337, mul_338, sub_112
#   relu_112 => relu_112
# Graph fragment:
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_53, %unsqueeze_897), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_901), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_903), kwargs = {})
#   %relu_112 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_225,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_90 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_90', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_90', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_90(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 896)
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


# kernel path: inductor_cache/vx/cvx6k6bmnhieheo5vppwanvqzg2pocbnz4kyg4qfhnpcure7c6i3.py
# Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_55 => cat_54
#   concated_features_56 => cat_55
#   concated_features_57 => cat_56
#   input_20 => cat_57
# Graph fragment:
#   %cat_54 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113], 1), kwargs = {})
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115], 1), kwargs = {})
#   %cat_56 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117], 1), kwargs = {})
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117, %convolution_119], 1), kwargs = {})
triton_poi_fused_cat_91 = async_compile.triton('triton_poi_fused_cat_91', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_91', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_91(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3712*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3840*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 3968*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 4096*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cuk2utu57whoie7qslpg46pzsvcepwp74hg3apjfro6pbwvzsu2u.py
# Topologically Sorted Source Nodes: [batch_norm_114, relu_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_114 => add_229, mul_343, mul_344, sub_114
#   relu_114 => relu_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_54, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %relu_114 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_92 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_92', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_92', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_92(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 928)
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


# kernel path: inductor_cache/ll/clllsqf5ogqomv2uf2l2fwgjuil3p5cc5to5pgzpkxxjim3mx5qr.py
# Topologically Sorted Source Nodes: [concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_56 => cat_55
#   concated_features_57 => cat_56
#   input_20 => cat_57
# Graph fragment:
#   %cat_55 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115], 1), kwargs = {})
#   %cat_56 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117], 1), kwargs = {})
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117, %convolution_119], 1), kwargs = {})
triton_poi_fused_cat_93 = async_compile.triton('triton_poi_fused_cat_93', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_93', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_93(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3840*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 3968*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 4096*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvum5epna3t6heq5mpphad4uzxz7pw3ssbkrwdpokztx7ex3dsyu.py
# Topologically Sorted Source Nodes: [batch_norm_116, relu_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_116 => add_233, mul_349, mul_350, sub_116
#   relu_116 => relu_116
# Graph fragment:
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_55, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
#   %relu_116 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_233,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_94 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_94', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_94', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_94(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 960)
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


# kernel path: inductor_cache/oo/coousep54y6mzcyhjn55wu5xiu4lkvcyabk3iitnwxhyyh4eo3mf.py
# Topologically Sorted Source Nodes: [concated_features_57, input_20], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_57 => cat_56
#   input_20 => cat_57
# Graph fragment:
#   %cat_56 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117], 1), kwargs = {})
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117, %convolution_119], 1), kwargs = {})
triton_poi_fused_cat_95 = async_compile.triton('triton_poi_fused_cat_95', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_95', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_95(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 3968*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 4096*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cji3nonixbcjrpshrhe7afb3pxzarxtnykdohcxpwkqt6kfew7t4.py
# Topologically Sorted Source Nodes: [batch_norm_118, relu_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_118 => add_237, mul_355, mul_356, sub_118
#   relu_118 => relu_118
# Graph fragment:
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_56, %unsqueeze_945), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_949), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_951), kwargs = {})
#   %relu_118 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_96 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_96', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_96', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_96(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 992)
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


# kernel path: inductor_cache/ky/ckyoy37vahr6ywpoksexgvms4myqg4fbclh3wepirul62t6x2ldy.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_20 => cat_57
# Graph fragment:
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_2, %convolution_89, %convolution_91, %convolution_93, %convolution_95, %convolution_97, %convolution_99, %convolution_101, %convolution_103, %convolution_105, %convolution_107, %convolution_109, %convolution_111, %convolution_113, %convolution_115, %convolution_117, %convolution_119], 1), kwargs = {})
triton_poi_fused_cat_97 = async_compile.triton('triton_poi_fused_cat_97', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_97', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_97(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 4096*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmjh2onxmyqqquqtyzq3mlpobab6lfqe5eeod6lxmjvvdeoma4ex.py
# Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
# Source node to ATen node mapping:
#   input_21 => add_241, mul_361, mul_362, sub_120
#   input_22 => mean
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_57, %unsqueeze_961), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_965), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_967), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_241, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_98 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_98', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_98', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_98(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (4*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (1 + 4*x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (2 + 4*x2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3 + 4*x2), None, eviction_policy='evict_last')
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
    tmp17 = tmp16 - tmp1
    tmp18 = tmp17 * tmp10
    tmp19 = tmp18 * tmp12
    tmp20 = tmp19 + tmp14
    tmp21 = tmp15 + tmp20
    tmp23 = tmp22 - tmp1
    tmp24 = tmp23 * tmp10
    tmp25 = tmp24 * tmp12
    tmp26 = tmp25 + tmp14
    tmp27 = tmp21 + tmp26
    tmp29 = tmp28 - tmp1
    tmp30 = tmp29 * tmp10
    tmp31 = tmp30 * tmp12
    tmp32 = tmp31 + tmp14
    tmp33 = tmp27 + tmp32
    tmp34 = 4.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (x2), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/wx/cwx7yrrly6uio4o7ae5dsmqspirt72sr6t5al7bzb42pbb7dj3ir.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_24 => add_242, add_243, mul_363, mul_364, mul_365, reciprocal_121, sqrt_121, sub_121
# Graph fragment:
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_609, 1e-05), kwargs = {})
#   %sqrt_121 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_242,), kwargs = {})
#   %reciprocal_121 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_121,), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_121, 1), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%addmm, %primals_608), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %mul_363), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_364, %primals_610), kwargs = {})
#   %add_243 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %primals_611), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_99 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_99', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_99', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_99(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613 = args
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
    assert_size_stride(primals_11, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_17, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_37, (160, ), (1, ))
    assert_size_stride(primals_38, (160, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_40, (160, ), (1, ))
    assert_size_stride(primals_41, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_57, (224, ), (1, ))
    assert_size_stride(primals_58, (224, ), (1, ))
    assert_size_stride(primals_59, (224, ), (1, ))
    assert_size_stride(primals_60, (224, ), (1, ))
    assert_size_stride(primals_61, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_82, (160, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (160, ), (1, ))
    assert_size_stride(primals_85, (160, ), (1, ))
    assert_size_stride(primals_86, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_92, (192, ), (1, ))
    assert_size_stride(primals_93, (192, ), (1, ))
    assert_size_stride(primals_94, (192, ), (1, ))
    assert_size_stride(primals_95, (192, ), (1, ))
    assert_size_stride(primals_96, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_102, (224, ), (1, ))
    assert_size_stride(primals_103, (224, ), (1, ))
    assert_size_stride(primals_104, (224, ), (1, ))
    assert_size_stride(primals_105, (224, ), (1, ))
    assert_size_stride(primals_106, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_122, (288, ), (1, ))
    assert_size_stride(primals_123, (288, ), (1, ))
    assert_size_stride(primals_124, (288, ), (1, ))
    assert_size_stride(primals_125, (288, ), (1, ))
    assert_size_stride(primals_126, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_132, (320, ), (1, ))
    assert_size_stride(primals_133, (320, ), (1, ))
    assert_size_stride(primals_134, (320, ), (1, ))
    assert_size_stride(primals_135, (320, ), (1, ))
    assert_size_stride(primals_136, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_142, (352, ), (1, ))
    assert_size_stride(primals_143, (352, ), (1, ))
    assert_size_stride(primals_144, (352, ), (1, ))
    assert_size_stride(primals_145, (352, ), (1, ))
    assert_size_stride(primals_146, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_162, (416, ), (1, ))
    assert_size_stride(primals_163, (416, ), (1, ))
    assert_size_stride(primals_164, (416, ), (1, ))
    assert_size_stride(primals_165, (416, ), (1, ))
    assert_size_stride(primals_166, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (448, ), (1, ))
    assert_size_stride(primals_173, (448, ), (1, ))
    assert_size_stride(primals_174, (448, ), (1, ))
    assert_size_stride(primals_175, (448, ), (1, ))
    assert_size_stride(primals_176, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_182, (480, ), (1, ))
    assert_size_stride(primals_183, (480, ), (1, ))
    assert_size_stride(primals_184, (480, ), (1, ))
    assert_size_stride(primals_185, (480, ), (1, ))
    assert_size_stride(primals_186, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_207, (288, ), (1, ))
    assert_size_stride(primals_208, (288, ), (1, ))
    assert_size_stride(primals_209, (288, ), (1, ))
    assert_size_stride(primals_210, (288, ), (1, ))
    assert_size_stride(primals_211, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, ), (1, ))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_217, (320, ), (1, ))
    assert_size_stride(primals_218, (320, ), (1, ))
    assert_size_stride(primals_219, (320, ), (1, ))
    assert_size_stride(primals_220, (320, ), (1, ))
    assert_size_stride(primals_221, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_227, (352, ), (1, ))
    assert_size_stride(primals_228, (352, ), (1, ))
    assert_size_stride(primals_229, (352, ), (1, ))
    assert_size_stride(primals_230, (352, ), (1, ))
    assert_size_stride(primals_231, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_242, (128, ), (1, ))
    assert_size_stride(primals_243, (128, ), (1, ))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_247, (416, ), (1, ))
    assert_size_stride(primals_248, (416, ), (1, ))
    assert_size_stride(primals_249, (416, ), (1, ))
    assert_size_stride(primals_250, (416, ), (1, ))
    assert_size_stride(primals_251, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_256, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_257, (448, ), (1, ))
    assert_size_stride(primals_258, (448, ), (1, ))
    assert_size_stride(primals_259, (448, ), (1, ))
    assert_size_stride(primals_260, (448, ), (1, ))
    assert_size_stride(primals_261, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_267, (480, ), (1, ))
    assert_size_stride(primals_268, (480, ), (1, ))
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_270, (480, ), (1, ))
    assert_size_stride(primals_271, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_272, (128, ), (1, ))
    assert_size_stride(primals_273, (128, ), (1, ))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, ), (1, ))
    assert_size_stride(primals_286, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_287, (544, ), (1, ))
    assert_size_stride(primals_288, (544, ), (1, ))
    assert_size_stride(primals_289, (544, ), (1, ))
    assert_size_stride(primals_290, (544, ), (1, ))
    assert_size_stride(primals_291, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_297, (576, ), (1, ))
    assert_size_stride(primals_298, (576, ), (1, ))
    assert_size_stride(primals_299, (576, ), (1, ))
    assert_size_stride(primals_300, (576, ), (1, ))
    assert_size_stride(primals_301, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_307, (608, ), (1, ))
    assert_size_stride(primals_308, (608, ), (1, ))
    assert_size_stride(primals_309, (608, ), (1, ))
    assert_size_stride(primals_310, (608, ), (1, ))
    assert_size_stride(primals_311, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_317, (640, ), (1, ))
    assert_size_stride(primals_318, (640, ), (1, ))
    assert_size_stride(primals_319, (640, ), (1, ))
    assert_size_stride(primals_320, (640, ), (1, ))
    assert_size_stride(primals_321, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_322, (128, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_326, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_327, (672, ), (1, ))
    assert_size_stride(primals_328, (672, ), (1, ))
    assert_size_stride(primals_329, (672, ), (1, ))
    assert_size_stride(primals_330, (672, ), (1, ))
    assert_size_stride(primals_331, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_337, (704, ), (1, ))
    assert_size_stride(primals_338, (704, ), (1, ))
    assert_size_stride(primals_339, (704, ), (1, ))
    assert_size_stride(primals_340, (704, ), (1, ))
    assert_size_stride(primals_341, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (128, ), (1, ))
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_347, (736, ), (1, ))
    assert_size_stride(primals_348, (736, ), (1, ))
    assert_size_stride(primals_349, (736, ), (1, ))
    assert_size_stride(primals_350, (736, ), (1, ))
    assert_size_stride(primals_351, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_352, (128, ), (1, ))
    assert_size_stride(primals_353, (128, ), (1, ))
    assert_size_stride(primals_354, (128, ), (1, ))
    assert_size_stride(primals_355, (128, ), (1, ))
    assert_size_stride(primals_356, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_357, (768, ), (1, ))
    assert_size_stride(primals_358, (768, ), (1, ))
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_362, (128, ), (1, ))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_367, (800, ), (1, ))
    assert_size_stride(primals_368, (800, ), (1, ))
    assert_size_stride(primals_369, (800, ), (1, ))
    assert_size_stride(primals_370, (800, ), (1, ))
    assert_size_stride(primals_371, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (128, ), (1, ))
    assert_size_stride(primals_374, (128, ), (1, ))
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_377, (832, ), (1, ))
    assert_size_stride(primals_378, (832, ), (1, ))
    assert_size_stride(primals_379, (832, ), (1, ))
    assert_size_stride(primals_380, (832, ), (1, ))
    assert_size_stride(primals_381, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (128, ), (1, ))
    assert_size_stride(primals_386, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_387, (864, ), (1, ))
    assert_size_stride(primals_388, (864, ), (1, ))
    assert_size_stride(primals_389, (864, ), (1, ))
    assert_size_stride(primals_390, (864, ), (1, ))
    assert_size_stride(primals_391, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_392, (128, ), (1, ))
    assert_size_stride(primals_393, (128, ), (1, ))
    assert_size_stride(primals_394, (128, ), (1, ))
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_397, (896, ), (1, ))
    assert_size_stride(primals_398, (896, ), (1, ))
    assert_size_stride(primals_399, (896, ), (1, ))
    assert_size_stride(primals_400, (896, ), (1, ))
    assert_size_stride(primals_401, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_403, (128, ), (1, ))
    assert_size_stride(primals_404, (128, ), (1, ))
    assert_size_stride(primals_405, (128, ), (1, ))
    assert_size_stride(primals_406, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_407, (928, ), (1, ))
    assert_size_stride(primals_408, (928, ), (1, ))
    assert_size_stride(primals_409, (928, ), (1, ))
    assert_size_stride(primals_410, (928, ), (1, ))
    assert_size_stride(primals_411, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_412, (128, ), (1, ))
    assert_size_stride(primals_413, (128, ), (1, ))
    assert_size_stride(primals_414, (128, ), (1, ))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_417, (960, ), (1, ))
    assert_size_stride(primals_418, (960, ), (1, ))
    assert_size_stride(primals_419, (960, ), (1, ))
    assert_size_stride(primals_420, (960, ), (1, ))
    assert_size_stride(primals_421, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, ), (1, ))
    assert_size_stride(primals_425, (128, ), (1, ))
    assert_size_stride(primals_426, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_427, (992, ), (1, ))
    assert_size_stride(primals_428, (992, ), (1, ))
    assert_size_stride(primals_429, (992, ), (1, ))
    assert_size_stride(primals_430, (992, ), (1, ))
    assert_size_stride(primals_431, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_437, (1024, ), (1, ))
    assert_size_stride(primals_438, (1024, ), (1, ))
    assert_size_stride(primals_439, (1024, ), (1, ))
    assert_size_stride(primals_440, (1024, ), (1, ))
    assert_size_stride(primals_441, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_442, (512, ), (1, ))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_444, (512, ), (1, ))
    assert_size_stride(primals_445, (512, ), (1, ))
    assert_size_stride(primals_446, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, ), (1, ))
    assert_size_stride(primals_451, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_452, (544, ), (1, ))
    assert_size_stride(primals_453, (544, ), (1, ))
    assert_size_stride(primals_454, (544, ), (1, ))
    assert_size_stride(primals_455, (544, ), (1, ))
    assert_size_stride(primals_456, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_457, (128, ), (1, ))
    assert_size_stride(primals_458, (128, ), (1, ))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_462, (576, ), (1, ))
    assert_size_stride(primals_463, (576, ), (1, ))
    assert_size_stride(primals_464, (576, ), (1, ))
    assert_size_stride(primals_465, (576, ), (1, ))
    assert_size_stride(primals_466, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_467, (128, ), (1, ))
    assert_size_stride(primals_468, (128, ), (1, ))
    assert_size_stride(primals_469, (128, ), (1, ))
    assert_size_stride(primals_470, (128, ), (1, ))
    assert_size_stride(primals_471, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_472, (608, ), (1, ))
    assert_size_stride(primals_473, (608, ), (1, ))
    assert_size_stride(primals_474, (608, ), (1, ))
    assert_size_stride(primals_475, (608, ), (1, ))
    assert_size_stride(primals_476, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_477, (128, ), (1, ))
    assert_size_stride(primals_478, (128, ), (1, ))
    assert_size_stride(primals_479, (128, ), (1, ))
    assert_size_stride(primals_480, (128, ), (1, ))
    assert_size_stride(primals_481, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_482, (640, ), (1, ))
    assert_size_stride(primals_483, (640, ), (1, ))
    assert_size_stride(primals_484, (640, ), (1, ))
    assert_size_stride(primals_485, (640, ), (1, ))
    assert_size_stride(primals_486, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_487, (128, ), (1, ))
    assert_size_stride(primals_488, (128, ), (1, ))
    assert_size_stride(primals_489, (128, ), (1, ))
    assert_size_stride(primals_490, (128, ), (1, ))
    assert_size_stride(primals_491, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_492, (672, ), (1, ))
    assert_size_stride(primals_493, (672, ), (1, ))
    assert_size_stride(primals_494, (672, ), (1, ))
    assert_size_stride(primals_495, (672, ), (1, ))
    assert_size_stride(primals_496, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (128, ), (1, ))
    assert_size_stride(primals_500, (128, ), (1, ))
    assert_size_stride(primals_501, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_502, (704, ), (1, ))
    assert_size_stride(primals_503, (704, ), (1, ))
    assert_size_stride(primals_504, (704, ), (1, ))
    assert_size_stride(primals_505, (704, ), (1, ))
    assert_size_stride(primals_506, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_507, (128, ), (1, ))
    assert_size_stride(primals_508, (128, ), (1, ))
    assert_size_stride(primals_509, (128, ), (1, ))
    assert_size_stride(primals_510, (128, ), (1, ))
    assert_size_stride(primals_511, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_512, (736, ), (1, ))
    assert_size_stride(primals_513, (736, ), (1, ))
    assert_size_stride(primals_514, (736, ), (1, ))
    assert_size_stride(primals_515, (736, ), (1, ))
    assert_size_stride(primals_516, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_517, (128, ), (1, ))
    assert_size_stride(primals_518, (128, ), (1, ))
    assert_size_stride(primals_519, (128, ), (1, ))
    assert_size_stride(primals_520, (128, ), (1, ))
    assert_size_stride(primals_521, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_522, (768, ), (1, ))
    assert_size_stride(primals_523, (768, ), (1, ))
    assert_size_stride(primals_524, (768, ), (1, ))
    assert_size_stride(primals_525, (768, ), (1, ))
    assert_size_stride(primals_526, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_527, (128, ), (1, ))
    assert_size_stride(primals_528, (128, ), (1, ))
    assert_size_stride(primals_529, (128, ), (1, ))
    assert_size_stride(primals_530, (128, ), (1, ))
    assert_size_stride(primals_531, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_532, (800, ), (1, ))
    assert_size_stride(primals_533, (800, ), (1, ))
    assert_size_stride(primals_534, (800, ), (1, ))
    assert_size_stride(primals_535, (800, ), (1, ))
    assert_size_stride(primals_536, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_537, (128, ), (1, ))
    assert_size_stride(primals_538, (128, ), (1, ))
    assert_size_stride(primals_539, (128, ), (1, ))
    assert_size_stride(primals_540, (128, ), (1, ))
    assert_size_stride(primals_541, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_542, (832, ), (1, ))
    assert_size_stride(primals_543, (832, ), (1, ))
    assert_size_stride(primals_544, (832, ), (1, ))
    assert_size_stride(primals_545, (832, ), (1, ))
    assert_size_stride(primals_546, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_547, (128, ), (1, ))
    assert_size_stride(primals_548, (128, ), (1, ))
    assert_size_stride(primals_549, (128, ), (1, ))
    assert_size_stride(primals_550, (128, ), (1, ))
    assert_size_stride(primals_551, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_552, (864, ), (1, ))
    assert_size_stride(primals_553, (864, ), (1, ))
    assert_size_stride(primals_554, (864, ), (1, ))
    assert_size_stride(primals_555, (864, ), (1, ))
    assert_size_stride(primals_556, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_557, (128, ), (1, ))
    assert_size_stride(primals_558, (128, ), (1, ))
    assert_size_stride(primals_559, (128, ), (1, ))
    assert_size_stride(primals_560, (128, ), (1, ))
    assert_size_stride(primals_561, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_562, (896, ), (1, ))
    assert_size_stride(primals_563, (896, ), (1, ))
    assert_size_stride(primals_564, (896, ), (1, ))
    assert_size_stride(primals_565, (896, ), (1, ))
    assert_size_stride(primals_566, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_567, (128, ), (1, ))
    assert_size_stride(primals_568, (128, ), (1, ))
    assert_size_stride(primals_569, (128, ), (1, ))
    assert_size_stride(primals_570, (128, ), (1, ))
    assert_size_stride(primals_571, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_572, (928, ), (1, ))
    assert_size_stride(primals_573, (928, ), (1, ))
    assert_size_stride(primals_574, (928, ), (1, ))
    assert_size_stride(primals_575, (928, ), (1, ))
    assert_size_stride(primals_576, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_577, (128, ), (1, ))
    assert_size_stride(primals_578, (128, ), (1, ))
    assert_size_stride(primals_579, (128, ), (1, ))
    assert_size_stride(primals_580, (128, ), (1, ))
    assert_size_stride(primals_581, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_582, (960, ), (1, ))
    assert_size_stride(primals_583, (960, ), (1, ))
    assert_size_stride(primals_584, (960, ), (1, ))
    assert_size_stride(primals_585, (960, ), (1, ))
    assert_size_stride(primals_586, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_587, (128, ), (1, ))
    assert_size_stride(primals_588, (128, ), (1, ))
    assert_size_stride(primals_589, (128, ), (1, ))
    assert_size_stride(primals_590, (128, ), (1, ))
    assert_size_stride(primals_591, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_592, (992, ), (1, ))
    assert_size_stride(primals_593, (992, ), (1, ))
    assert_size_stride(primals_594, (992, ), (1, ))
    assert_size_stride(primals_595, (992, ), (1, ))
    assert_size_stride(primals_596, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_597, (128, ), (1, ))
    assert_size_stride(primals_598, (128, ), (1, ))
    assert_size_stride(primals_599, (128, ), (1, ))
    assert_size_stride(primals_600, (128, ), (1, ))
    assert_size_stride(primals_601, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_602, (1024, ), (1, ))
    assert_size_stride(primals_603, (1024, ), (1, ))
    assert_size_stride(primals_604, (1024, ), (1, ))
    assert_size_stride(primals_605, (1024, ), (1, ))
    assert_size_stride(primals_606, (512, 1024), (1024, 1))
    assert_size_stride(primals_607, (512, ), (1, ))
    assert_size_stride(primals_608, (512, ), (1, ))
    assert_size_stride(primals_609, (512, ), (1, ))
    assert_size_stride(primals_610, (512, ), (1, ))
    assert_size_stride(primals_611, (512, ), (1, ))
    assert_size_stride(primals_612, (4, 512), (512, 1))
    assert_size_stride(primals_613, (4, ), (1, ))
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
        buf9 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        buf2 = reinterpret_tensor(buf9, (4, 64, 16, 16), (24576, 256, 16, 1), 0)  # alias
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf769 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, batch_norm_1, relu_1], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1.run(buf1, primals_7, primals_8, primals_9, primals_10, buf2, buf3, buf4, buf769, 65536, grid=grid(65536), stream=stream0)
        del primals_10
        del primals_7
        # Topologically Sorted Source Nodes: [bottleneck_output], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf6 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_2, relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, primals_12, primals_13, primals_14, primals_15, buf6, 131072, grid=grid(131072), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [new_features], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf8 = reinterpret_tensor(buf9, (4, 32, 16, 16), (24576, 256, 16, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [concated_features_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf7, buf8, 32768, grid=grid(32768), stream=stream0)
        buf10 = empty_strided_cuda((4, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, primals_17, primals_18, primals_19, primals_20, buf10, 98304, grid=grid(98304), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [bottleneck_output_1], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf12 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_4, relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf11, primals_22, primals_23, primals_24, primals_25, buf12, 131072, grid=grid(131072), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [new_features_1], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_2, batch_norm_5, relu_5], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf2, buf7, buf13, primals_27, primals_28, primals_29, primals_30, buf14, buf15, 131072, grid=grid(131072), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [bottleneck_output_2], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_6, relu_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf16, primals_32, primals_33, primals_34, primals_35, buf17, 131072, grid=grid(131072), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [new_features_2], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 160, 16, 16), (40960, 256, 16, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 160, 16, 16), (40960, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_3, batch_norm_7, relu_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf2, buf7, buf13, buf18, primals_37, primals_38, primals_39, primals_40, buf19, buf20, 163840, grid=grid(163840), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [bottleneck_output_3], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf22 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_8, relu_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf21, primals_42, primals_43, primals_44, primals_45, buf22, 131072, grid=grid(131072), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [new_features_3], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf24 = empty_strided_cuda((4, 192, 16, 16), (49152, 256, 16, 1), torch.float32)
        buf25 = empty_strided_cuda((4, 192, 16, 16), (49152, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_4, batch_norm_9, relu_9], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7.run(buf2, buf7, buf13, buf18, buf23, primals_47, primals_48, primals_49, primals_50, buf24, buf25, 196608, grid=grid(196608), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [bottleneck_output_4], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf27 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_10, relu_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf26, primals_52, primals_53, primals_54, primals_55, buf27, 131072, grid=grid(131072), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [new_features_4], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf29 = empty_strided_cuda((4, 224, 16, 16), (57344, 256, 16, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 224, 16, 16), (57344, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_5, batch_norm_11, relu_11], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8.run(buf2, buf7, buf13, buf18, buf23, buf28, primals_57, primals_58, primals_59, primals_60, buf29, buf30, 229376, grid=grid(229376), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [bottleneck_output_5], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf32 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_12, relu_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf31, primals_62, primals_63, primals_64, primals_65, buf32, 131072, grid=grid(131072), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [new_features_5], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf34 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf35 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf2, buf7, buf13, buf18, buf23, buf28, buf33, primals_67, primals_68, primals_69, primals_70, buf34, buf35, 262144, grid=grid(262144), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf37 = reinterpret_tensor(buf7, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf7  # reuse
        buf38 = reinterpret_tensor(buf33, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf33  # reuse
        buf86 = empty_strided_cuda((4, 384, 8, 8), (24576, 64, 8, 1), torch.float32)
        buf77 = reinterpret_tensor(buf86, (4, 128, 8, 8), (24576, 64, 8, 1), 0)  # alias
        buf101 = empty_strided_cuda((4, 416, 8, 8), (26624, 64, 8, 1), torch.float32)
        buf91 = reinterpret_tensor(buf101, (4, 128, 8, 8), (26624, 64, 8, 1), 0)  # alias
        buf117 = empty_strided_cuda((4, 448, 8, 8), (28672, 64, 8, 1), torch.float32)
        buf106 = reinterpret_tensor(buf117, (4, 128, 8, 8), (28672, 64, 8, 1), 0)  # alias
        buf134 = empty_strided_cuda((4, 480, 8, 8), (30720, 64, 8, 1), torch.float32)
        buf122 = reinterpret_tensor(buf134, (4, 128, 8, 8), (30720, 64, 8, 1), 0)  # alias
        buf152 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf139 = reinterpret_tensor(buf152, (4, 128, 8, 8), (32768, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_9, batch_norm_14, relu_14, concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_10.run(buf36, primals_72, primals_73, primals_74, primals_75, buf37, buf38, buf77, buf91, buf106, buf122, buf139, 32768, grid=grid(32768), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [bottleneck_output_6], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf40 = reinterpret_tensor(buf28, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_15, relu_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf39, primals_77, primals_78, primals_79, primals_80, buf40, 32768, grid=grid(32768), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [new_features_6], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf42 = empty_strided_cuda((4, 160, 8, 8), (10240, 64, 8, 1), torch.float32)
        buf43 = empty_strided_cuda((4, 160, 8, 8), (10240, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_7, batch_norm_16, relu_16], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf37, buf41, primals_82, primals_83, primals_84, primals_85, buf42, buf43, 40960, grid=grid(40960), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [bottleneck_output_7], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf45 = reinterpret_tensor(buf23, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_17, relu_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf44, primals_87, primals_88, primals_89, primals_90, buf45, 32768, grid=grid(32768), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [new_features_7], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf47 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        buf48 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_8, batch_norm_18, relu_18], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13.run(buf37, buf41, buf46, primals_92, primals_93, primals_94, primals_95, buf47, buf48, 49152, grid=grid(49152), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [bottleneck_output_8], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf50 = reinterpret_tensor(buf18, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_19, relu_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf49, primals_97, primals_98, primals_99, primals_100, buf50, 32768, grid=grid(32768), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [new_features_8], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf52 = empty_strided_cuda((4, 224, 8, 8), (14336, 64, 8, 1), torch.float32)
        buf53 = empty_strided_cuda((4, 224, 8, 8), (14336, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_9, batch_norm_20, relu_20], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14.run(buf37, buf41, buf46, buf51, primals_102, primals_103, primals_104, primals_105, buf52, buf53, 57344, grid=grid(57344), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [bottleneck_output_9], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf55 = reinterpret_tensor(buf13, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_21, relu_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf54, primals_107, primals_108, primals_109, primals_110, buf55, 32768, grid=grid(32768), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [new_features_9], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf57 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf58 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_10, batch_norm_22, relu_22], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15.run(buf37, buf41, buf46, buf51, buf56, primals_112, primals_113, primals_114, primals_115, buf57, buf58, 65536, grid=grid(65536), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [bottleneck_output_10], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf60 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_23, relu_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf59, primals_117, primals_118, primals_119, primals_120, buf60, 32768, grid=grid(32768), stream=stream0)
        del primals_120
        # Topologically Sorted Source Nodes: [new_features_10], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf62 = empty_strided_cuda((4, 288, 8, 8), (18432, 64, 8, 1), torch.float32)
        buf63 = empty_strided_cuda((4, 288, 8, 8), (18432, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_11, batch_norm_24, relu_24], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf37, buf41, buf46, buf51, buf56, buf61, primals_122, primals_123, primals_124, primals_125, buf62, buf63, 73728, grid=grid(73728), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [bottleneck_output_11], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf65 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_25, relu_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf64, primals_127, primals_128, primals_129, primals_130, buf65, 32768, grid=grid(32768), stream=stream0)
        del primals_130
        # Topologically Sorted Source Nodes: [new_features_11], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf67 = empty_strided_cuda((4, 320, 8, 8), (20480, 64, 8, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 320, 8, 8), (20480, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_12, batch_norm_26, relu_26], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17.run(buf37, buf41, buf46, buf51, buf56, buf61, buf66, primals_132, primals_133, primals_134, primals_135, buf67, buf68, 81920, grid=grid(81920), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [bottleneck_output_12], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf70 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_27, relu_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf69, primals_137, primals_138, primals_139, primals_140, buf70, 32768, grid=grid(32768), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [new_features_12], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf72 = empty_strided_cuda((4, 352, 8, 8), (22528, 64, 8, 1), torch.float32)
        buf73 = empty_strided_cuda((4, 352, 8, 8), (22528, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_13, batch_norm_28, relu_28], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18.run(buf37, buf41, buf46, buf51, buf56, buf61, buf66, buf71, primals_142, primals_143, primals_144, primals_145, buf72, buf73, 90112, grid=grid(90112), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [bottleneck_output_13], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf75 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_29, relu_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf74, primals_147, primals_148, primals_149, primals_150, buf75, 32768, grid=grid(32768), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [new_features_13], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf78 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 8192)  # alias
        buf92 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 8192)  # alias
        buf107 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 8192)  # alias
        buf123 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 8192)  # alias
        buf140 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf41, buf78, buf92, buf107, buf123, buf140, 8192, grid=grid(8192), stream=stream0)
        buf79 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 10240)  # alias
        buf93 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 10240)  # alias
        buf108 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 10240)  # alias
        buf124 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 10240)  # alias
        buf141 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf46, buf79, buf93, buf108, buf124, buf141, 8192, grid=grid(8192), stream=stream0)
        buf80 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 12288)  # alias
        buf94 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 12288)  # alias
        buf109 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 12288)  # alias
        buf125 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 12288)  # alias
        buf142 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf51, buf80, buf94, buf109, buf125, buf142, 8192, grid=grid(8192), stream=stream0)
        buf81 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 14336)  # alias
        buf95 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 14336)  # alias
        buf110 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 14336)  # alias
        buf126 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 14336)  # alias
        buf143 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 14336)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf56, buf81, buf95, buf110, buf126, buf143, 8192, grid=grid(8192), stream=stream0)
        buf82 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 16384)  # alias
        buf96 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 16384)  # alias
        buf111 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 16384)  # alias
        buf127 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 16384)  # alias
        buf144 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf61, buf82, buf96, buf111, buf127, buf144, 8192, grid=grid(8192), stream=stream0)
        buf83 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 18432)  # alias
        buf97 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 18432)  # alias
        buf112 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 18432)  # alias
        buf128 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 18432)  # alias
        buf145 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 18432)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf66, buf83, buf97, buf112, buf128, buf145, 8192, grid=grid(8192), stream=stream0)
        buf84 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 20480)  # alias
        buf98 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 20480)  # alias
        buf113 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 20480)  # alias
        buf129 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 20480)  # alias
        buf146 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf71, buf84, buf98, buf113, buf129, buf146, 8192, grid=grid(8192), stream=stream0)
        buf85 = reinterpret_tensor(buf86, (4, 32, 8, 8), (24576, 64, 8, 1), 22528)  # alias
        buf99 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 22528)  # alias
        buf114 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 22528)  # alias
        buf130 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 22528)  # alias
        buf147 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 22528)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf76, buf85, buf99, buf114, buf130, buf147, 8192, grid=grid(8192), stream=stream0)
        buf87 = empty_strided_cuda((4, 384, 8, 8), (24576, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_30, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf86, primals_152, primals_153, primals_154, primals_155, buf87, 98304, grid=grid(98304), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [bottleneck_output_14], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf89 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_31, relu_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf88, primals_157, primals_158, primals_159, primals_160, buf89, 32768, grid=grid(32768), stream=stream0)
        del primals_160
        # Topologically Sorted Source Nodes: [new_features_14], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf100 = reinterpret_tensor(buf101, (4, 32, 8, 8), (26624, 64, 8, 1), 24576)  # alias
        buf115 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 24576)  # alias
        buf131 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 24576)  # alias
        buf148 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 24576)  # alias
        # Topologically Sorted Source Nodes: [concated_features_15, concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf90, buf100, buf115, buf131, buf148, 8192, grid=grid(8192), stream=stream0)
        buf102 = empty_strided_cuda((4, 416, 8, 8), (26624, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_32, relu_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf101, primals_162, primals_163, primals_164, primals_165, buf102, 106496, grid=grid(106496), stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [bottleneck_output_15], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf104 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_33, relu_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf103, primals_167, primals_168, primals_169, primals_170, buf104, 32768, grid=grid(32768), stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [new_features_15], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf116 = reinterpret_tensor(buf117, (4, 32, 8, 8), (28672, 64, 8, 1), 26624)  # alias
        buf132 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 26624)  # alias
        buf149 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 26624)  # alias
        # Topologically Sorted Source Nodes: [concated_features_16, concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf105, buf116, buf132, buf149, 8192, grid=grid(8192), stream=stream0)
        buf118 = empty_strided_cuda((4, 448, 8, 8), (28672, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf117, primals_172, primals_173, primals_174, primals_175, buf118, 114688, grid=grid(114688), stream=stream0)
        del primals_175
        # Topologically Sorted Source Nodes: [bottleneck_output_16], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf120 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_35, relu_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf119, primals_177, primals_178, primals_179, primals_180, buf120, 32768, grid=grid(32768), stream=stream0)
        del primals_180
        # Topologically Sorted Source Nodes: [new_features_16], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf133 = reinterpret_tensor(buf134, (4, 32, 8, 8), (30720, 64, 8, 1), 28672)  # alias
        buf150 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 28672)  # alias
        # Topologically Sorted Source Nodes: [concated_features_17, input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf121, buf133, buf150, 8192, grid=grid(8192), stream=stream0)
        buf135 = empty_strided_cuda((4, 480, 8, 8), (30720, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_36, relu_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf134, primals_182, primals_183, primals_184, primals_185, buf135, 122880, grid=grid(122880), stream=stream0)
        del primals_185
        # Topologically Sorted Source Nodes: [bottleneck_output_17], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf137 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_37, relu_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf136, primals_187, primals_188, primals_189, primals_190, buf137, 32768, grid=grid(32768), stream=stream0)
        del primals_190
        # Topologically Sorted Source Nodes: [new_features_17], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf151 = reinterpret_tensor(buf152, (4, 32, 8, 8), (32768, 64, 8, 1), 30720)  # alias
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf138, buf151, 8192, grid=grid(8192), stream=stream0)
        buf153 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf152, primals_192, primals_193, primals_194, primals_195, buf153, 131072, grid=grid(131072), stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf155 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf156 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf204 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf195 = reinterpret_tensor(buf204, (4, 256, 4, 4), (8192, 16, 4, 1), 0)  # alias
        buf219 = empty_strided_cuda((4, 544, 4, 4), (8704, 16, 4, 1), torch.float32)
        buf209 = reinterpret_tensor(buf219, (4, 256, 4, 4), (8704, 16, 4, 1), 0)  # alias
        buf235 = empty_strided_cuda((4, 576, 4, 4), (9216, 16, 4, 1), torch.float32)
        buf224 = reinterpret_tensor(buf235, (4, 256, 4, 4), (9216, 16, 4, 1), 0)  # alias
        buf252 = empty_strided_cuda((4, 608, 4, 4), (9728, 16, 4, 1), torch.float32)
        buf240 = reinterpret_tensor(buf252, (4, 256, 4, 4), (9728, 16, 4, 1), 0)  # alias
        buf270 = empty_strided_cuda((4, 640, 4, 4), (10240, 16, 4, 1), torch.float32)
        buf257 = reinterpret_tensor(buf270, (4, 256, 4, 4), (10240, 16, 4, 1), 0)  # alias
        buf289 = empty_strided_cuda((4, 672, 4, 4), (10752, 16, 4, 1), torch.float32)
        buf275 = reinterpret_tensor(buf289, (4, 256, 4, 4), (10752, 16, 4, 1), 0)  # alias
        buf309 = empty_strided_cuda((4, 704, 4, 4), (11264, 16, 4, 1), torch.float32)
        buf294 = reinterpret_tensor(buf309, (4, 256, 4, 4), (11264, 16, 4, 1), 0)  # alias
        buf330 = empty_strided_cuda((4, 736, 4, 4), (11776, 16, 4, 1), torch.float32)
        buf314 = reinterpret_tensor(buf330, (4, 256, 4, 4), (11776, 16, 4, 1), 0)  # alias
        buf352 = empty_strided_cuda((4, 768, 4, 4), (12288, 16, 4, 1), torch.float32)
        buf335 = reinterpret_tensor(buf352, (4, 256, 4, 4), (12288, 16, 4, 1), 0)  # alias
        buf375 = empty_strided_cuda((4, 800, 4, 4), (12800, 16, 4, 1), torch.float32)
        buf357 = reinterpret_tensor(buf375, (4, 256, 4, 4), (12800, 16, 4, 1), 0)  # alias
        buf399 = empty_strided_cuda((4, 832, 4, 4), (13312, 16, 4, 1), torch.float32)
        buf380 = reinterpret_tensor(buf399, (4, 256, 4, 4), (13312, 16, 4, 1), 0)  # alias
        buf424 = empty_strided_cuda((4, 864, 4, 4), (13824, 16, 4, 1), torch.float32)
        buf404 = reinterpret_tensor(buf424, (4, 256, 4, 4), (13824, 16, 4, 1), 0)  # alias
        buf450 = empty_strided_cuda((4, 896, 4, 4), (14336, 16, 4, 1), torch.float32)
        buf429 = reinterpret_tensor(buf450, (4, 256, 4, 4), (14336, 16, 4, 1), 0)  # alias
        buf477 = empty_strided_cuda((4, 928, 4, 4), (14848, 16, 4, 1), torch.float32)
        buf455 = reinterpret_tensor(buf477, (4, 256, 4, 4), (14848, 16, 4, 1), 0)  # alias
        buf505 = empty_strided_cuda((4, 960, 4, 4), (15360, 16, 4, 1), torch.float32)
        buf482 = reinterpret_tensor(buf505, (4, 256, 4, 4), (15360, 16, 4, 1), 0)  # alias
        buf534 = empty_strided_cuda((4, 992, 4, 4), (15872, 16, 4, 1), torch.float32)
        buf510 = reinterpret_tensor(buf534, (4, 256, 4, 4), (15872, 16, 4, 1), 0)  # alias
        buf564 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf539 = reinterpret_tensor(buf564, (4, 256, 4, 4), (16384, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_14, batch_norm_39, relu_39, concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30, concated_features_31, concated_features_32, concated_features_33, concated_features_34, concated_features_35, concated_features_36, concated_features_37, concated_features_38, concated_features_39, concated_features_40, concated_features_41, input_15], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29.run(buf154, primals_197, primals_198, primals_199, primals_200, buf155, buf156, buf195, buf209, buf224, buf240, buf257, buf275, buf294, buf314, buf335, buf357, buf380, buf404, buf429, buf455, buf482, buf510, buf539, 16384, grid=grid(16384), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [bottleneck_output_18], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf158 = reinterpret_tensor(buf138, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_40, relu_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf157, primals_202, primals_203, primals_204, primals_205, buf158, 8192, grid=grid(8192), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [new_features_18], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 32, 4, 4), (512, 16, 4, 1))
        buf160 = empty_strided_cuda((4, 288, 4, 4), (4608, 16, 4, 1), torch.float32)
        buf161 = empty_strided_cuda((4, 288, 4, 4), (4608, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_19, batch_norm_41, relu_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31.run(buf155, buf159, primals_207, primals_208, primals_209, primals_210, buf160, buf161, 18432, grid=grid(18432), stream=stream0)
        del primals_210
        # Topologically Sorted Source Nodes: [bottleneck_output_19], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf163 = reinterpret_tensor(buf121, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_42, relu_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf162, primals_212, primals_213, primals_214, primals_215, buf163, 8192, grid=grid(8192), stream=stream0)
        del primals_215
        # Topologically Sorted Source Nodes: [new_features_19], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 32, 4, 4), (512, 16, 4, 1))
        buf165 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        buf166 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_20, batch_norm_43, relu_43], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32.run(buf155, buf159, buf164, primals_217, primals_218, primals_219, primals_220, buf165, buf166, 20480, grid=grid(20480), stream=stream0)
        del primals_220
        # Topologically Sorted Source Nodes: [bottleneck_output_20], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf168 = reinterpret_tensor(buf105, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_44, relu_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf167, primals_222, primals_223, primals_224, primals_225, buf168, 8192, grid=grid(8192), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [new_features_20], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 32, 4, 4), (512, 16, 4, 1))
        buf170 = empty_strided_cuda((4, 352, 4, 4), (5632, 16, 4, 1), torch.float32)
        buf171 = empty_strided_cuda((4, 352, 4, 4), (5632, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_21, batch_norm_45, relu_45], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33.run(buf155, buf159, buf164, buf169, primals_227, primals_228, primals_229, primals_230, buf170, buf171, 22528, grid=grid(22528), stream=stream0)
        del primals_230
        # Topologically Sorted Source Nodes: [bottleneck_output_21], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf173 = reinterpret_tensor(buf90, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_46, relu_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf172, primals_232, primals_233, primals_234, primals_235, buf173, 8192, grid=grid(8192), stream=stream0)
        del primals_235
        # Topologically Sorted Source Nodes: [new_features_21], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_236, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 32, 4, 4), (512, 16, 4, 1))
        buf175 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        buf176 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_22, batch_norm_47, relu_47], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34.run(buf155, buf159, buf164, buf169, buf174, primals_237, primals_238, primals_239, primals_240, buf175, buf176, 24576, grid=grid(24576), stream=stream0)
        del primals_240
        # Topologically Sorted Source Nodes: [bottleneck_output_22], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf178 = reinterpret_tensor(buf76, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_48, relu_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf177, primals_242, primals_243, primals_244, primals_245, buf178, 8192, grid=grid(8192), stream=stream0)
        del primals_245
        # Topologically Sorted Source Nodes: [new_features_22], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 32, 4, 4), (512, 16, 4, 1))
        buf180 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        buf181 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_23, batch_norm_49, relu_49], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35.run(buf155, buf159, buf164, buf169, buf174, buf179, primals_247, primals_248, primals_249, primals_250, buf180, buf181, 26624, grid=grid(26624), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [bottleneck_output_23], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf183 = reinterpret_tensor(buf71, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_50, relu_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf182, primals_252, primals_253, primals_254, primals_255, buf183, 8192, grid=grid(8192), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [new_features_23], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 32, 4, 4), (512, 16, 4, 1))
        buf185 = empty_strided_cuda((4, 448, 4, 4), (7168, 16, 4, 1), torch.float32)
        buf186 = empty_strided_cuda((4, 448, 4, 4), (7168, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_24, batch_norm_51, relu_51], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36.run(buf155, buf159, buf164, buf169, buf174, buf179, buf184, primals_257, primals_258, primals_259, primals_260, buf185, buf186, 28672, grid=grid(28672), stream=stream0)
        del primals_260
        # Topologically Sorted Source Nodes: [bottleneck_output_24], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf188 = reinterpret_tensor(buf66, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_52, relu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf187, primals_262, primals_263, primals_264, primals_265, buf188, 8192, grid=grid(8192), stream=stream0)
        del primals_265
        # Topologically Sorted Source Nodes: [new_features_24], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 32, 4, 4), (512, 16, 4, 1))
        buf190 = empty_strided_cuda((4, 480, 4, 4), (7680, 16, 4, 1), torch.float32)
        buf191 = empty_strided_cuda((4, 480, 4, 4), (7680, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_25, batch_norm_53, relu_53], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37.run(buf155, buf159, buf164, buf169, buf174, buf179, buf184, buf189, primals_267, primals_268, primals_269, primals_270, buf190, buf191, 30720, grid=grid(30720), stream=stream0)
        del primals_270
        # Topologically Sorted Source Nodes: [bottleneck_output_25], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf193 = reinterpret_tensor(buf61, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_54, relu_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf192, primals_272, primals_273, primals_274, primals_275, buf193, 8192, grid=grid(8192), stream=stream0)
        del primals_275
        # Topologically Sorted Source Nodes: [new_features_25], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 32, 4, 4), (512, 16, 4, 1))
        buf196 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 4096)  # alias
        buf210 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 4096)  # alias
        buf225 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 4096)  # alias
        buf241 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 4096)  # alias
        buf258 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf159, buf196, buf210, buf225, buf241, buf258, 2048, grid=grid(2048), stream=stream0)
        buf197 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 4608)  # alias
        buf211 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 4608)  # alias
        buf226 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 4608)  # alias
        buf242 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 4608)  # alias
        buf259 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 4608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf164, buf197, buf211, buf226, buf242, buf259, 2048, grid=grid(2048), stream=stream0)
        buf198 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 5120)  # alias
        buf212 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 5120)  # alias
        buf227 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 5120)  # alias
        buf243 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 5120)  # alias
        buf260 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf169, buf198, buf212, buf227, buf243, buf260, 2048, grid=grid(2048), stream=stream0)
        buf199 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 5632)  # alias
        buf213 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 5632)  # alias
        buf228 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 5632)  # alias
        buf244 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 5632)  # alias
        buf261 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 5632)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf174, buf199, buf213, buf228, buf244, buf261, 2048, grid=grid(2048), stream=stream0)
        buf200 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 6144)  # alias
        buf214 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 6144)  # alias
        buf229 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 6144)  # alias
        buf245 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 6144)  # alias
        buf262 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf179, buf200, buf214, buf229, buf245, buf262, 2048, grid=grid(2048), stream=stream0)
        buf201 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 6656)  # alias
        buf215 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 6656)  # alias
        buf230 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 6656)  # alias
        buf246 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 6656)  # alias
        buf263 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 6656)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf184, buf201, buf215, buf230, buf246, buf263, 2048, grid=grid(2048), stream=stream0)
        buf202 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 7168)  # alias
        buf216 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 7168)  # alias
        buf231 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 7168)  # alias
        buf247 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 7168)  # alias
        buf264 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 7168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf189, buf202, buf216, buf231, buf247, buf264, 2048, grid=grid(2048), stream=stream0)
        buf203 = reinterpret_tensor(buf204, (4, 32, 4, 4), (8192, 16, 4, 1), 7680)  # alias
        buf217 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 7680)  # alias
        buf232 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 7680)  # alias
        buf248 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 7680)  # alias
        buf265 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 7680)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf194, buf203, buf217, buf232, buf248, buf265, 2048, grid=grid(2048), stream=stream0)
        buf205 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_55, relu_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf204, primals_277, primals_278, primals_279, primals_280, buf205, 32768, grid=grid(32768), stream=stream0)
        del primals_280
        # Topologically Sorted Source Nodes: [bottleneck_output_26], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf207 = reinterpret_tensor(buf56, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_56, relu_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf206, primals_282, primals_283, primals_284, primals_285, buf207, 8192, grid=grid(8192), stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [new_features_26], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 32, 4, 4), (512, 16, 4, 1))
        buf218 = reinterpret_tensor(buf219, (4, 32, 4, 4), (8704, 16, 4, 1), 8192)  # alias
        buf233 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 8192)  # alias
        buf249 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 8192)  # alias
        buf266 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf208, buf218, buf233, buf249, buf266, 2048, grid=grid(2048), stream=stream0)
        buf220 = empty_strided_cuda((4, 544, 4, 4), (8704, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57, relu_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf219, primals_287, primals_288, primals_289, primals_290, buf220, 34816, grid=grid(34816), stream=stream0)
        del primals_290
        # Topologically Sorted Source Nodes: [bottleneck_output_27], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf222 = reinterpret_tensor(buf51, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_58, relu_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf221, primals_292, primals_293, primals_294, primals_295, buf222, 8192, grid=grid(8192), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [new_features_27], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_296, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 32, 4, 4), (512, 16, 4, 1))
        buf234 = reinterpret_tensor(buf235, (4, 32, 4, 4), (9216, 16, 4, 1), 8704)  # alias
        buf250 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 8704)  # alias
        buf267 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 8704)  # alias
        buf285 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 8704)  # alias
        # Topologically Sorted Source Nodes: [concated_features_28, concated_features_29, concated_features_30, concated_features_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_42.run(buf223, buf234, buf250, buf267, buf285, 2048, grid=grid(2048), stream=stream0)
        buf236 = empty_strided_cuda((4, 576, 4, 4), (9216, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59, relu_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf235, primals_297, primals_298, primals_299, primals_300, buf236, 36864, grid=grid(36864), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [bottleneck_output_28], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf238 = reinterpret_tensor(buf46, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_60, relu_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf237, primals_302, primals_303, primals_304, primals_305, buf238, 8192, grid=grid(8192), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [new_features_28], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 32, 4, 4), (512, 16, 4, 1))
        buf251 = reinterpret_tensor(buf252, (4, 32, 4, 4), (9728, 16, 4, 1), 9216)  # alias
        buf268 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 9216)  # alias
        buf286 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 9216)  # alias
        buf305 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 9216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, concated_features_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_44.run(buf239, buf251, buf268, buf286, buf305, 2048, grid=grid(2048), stream=stream0)
        buf253 = empty_strided_cuda((4, 608, 4, 4), (9728, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf252, primals_307, primals_308, primals_309, primals_310, buf253, 38912, grid=grid(38912), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [bottleneck_output_29], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf255 = reinterpret_tensor(buf41, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_62, relu_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf254, primals_312, primals_313, primals_314, primals_315, buf255, 8192, grid=grid(8192), stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [new_features_29], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 32, 4, 4), (512, 16, 4, 1))
        buf269 = reinterpret_tensor(buf270, (4, 32, 4, 4), (10240, 16, 4, 1), 9728)  # alias
        buf287 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 9728)  # alias
        buf306 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 9728)  # alias
        buf326 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 9728)  # alias
        # Topologically Sorted Source Nodes: [concated_features_30, concated_features_31, concated_features_32, concated_features_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_46.run(buf256, buf269, buf287, buf306, buf326, 2048, grid=grid(2048), stream=stream0)
        buf271 = empty_strided_cuda((4, 640, 4, 4), (10240, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63, relu_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf270, primals_317, primals_318, primals_319, primals_320, buf271, 40960, grid=grid(40960), stream=stream0)
        del primals_320
        # Topologically Sorted Source Nodes: [bottleneck_output_30], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf273 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_64, relu_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf272, primals_322, primals_323, primals_324, primals_325, buf273, 8192, grid=grid(8192), stream=stream0)
        del primals_325
        # Topologically Sorted Source Nodes: [new_features_30], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 32, 4, 4), (512, 16, 4, 1))
        buf276 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 4096)  # alias
        buf295 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 4096)  # alias
        buf315 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 4096)  # alias
        buf336 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf159, buf276, buf295, buf315, buf336, 2048, grid=grid(2048), stream=stream0)
        buf277 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 4608)  # alias
        buf296 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 4608)  # alias
        buf316 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 4608)  # alias
        buf337 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 4608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf164, buf277, buf296, buf316, buf337, 2048, grid=grid(2048), stream=stream0)
        buf278 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 5120)  # alias
        buf297 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 5120)  # alias
        buf317 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 5120)  # alias
        buf338 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf169, buf278, buf297, buf317, buf338, 2048, grid=grid(2048), stream=stream0)
        buf279 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 5632)  # alias
        buf298 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 5632)  # alias
        buf318 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 5632)  # alias
        buf339 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 5632)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf174, buf279, buf298, buf318, buf339, 2048, grid=grid(2048), stream=stream0)
        buf280 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 6144)  # alias
        buf299 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 6144)  # alias
        buf319 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 6144)  # alias
        buf340 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf179, buf280, buf299, buf319, buf340, 2048, grid=grid(2048), stream=stream0)
        buf281 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 6656)  # alias
        buf300 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 6656)  # alias
        buf320 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 6656)  # alias
        buf341 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 6656)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf184, buf281, buf300, buf320, buf341, 2048, grid=grid(2048), stream=stream0)
        buf282 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 7168)  # alias
        buf301 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 7168)  # alias
        buf321 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 7168)  # alias
        buf342 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 7168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf189, buf282, buf301, buf321, buf342, 2048, grid=grid(2048), stream=stream0)
        buf283 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 7680)  # alias
        buf302 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 7680)  # alias
        buf322 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 7680)  # alias
        buf343 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 7680)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf194, buf283, buf302, buf322, buf343, 2048, grid=grid(2048), stream=stream0)
        buf284 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 8192)  # alias
        buf303 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 8192)  # alias
        buf323 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 8192)  # alias
        buf344 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf208, buf284, buf303, buf323, buf344, 2048, grid=grid(2048), stream=stream0)
        buf288 = reinterpret_tensor(buf289, (4, 32, 4, 4), (10752, 16, 4, 1), 10240)  # alias
        buf307 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 10240)  # alias
        buf327 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 10240)  # alias
        buf348 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, concated_features_32, concated_features_33, concated_features_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf274, buf288, buf307, buf327, buf348, 2048, grid=grid(2048), stream=stream0)
        buf290 = empty_strided_cuda((4, 672, 4, 4), (10752, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_65, relu_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf289, primals_327, primals_328, primals_329, primals_330, buf290, 43008, grid=grid(43008), stream=stream0)
        del primals_330
        # Topologically Sorted Source Nodes: [bottleneck_output_31], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_331, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf292 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_66, relu_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf291, primals_332, primals_333, primals_334, primals_335, buf292, 8192, grid=grid(8192), stream=stream0)
        del primals_335
        # Topologically Sorted Source Nodes: [new_features_31], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_336, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 32, 4, 4), (512, 16, 4, 1))
        buf304 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 8704)  # alias
        buf324 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 8704)  # alias
        buf345 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 8704)  # alias
        buf367 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 8704)  # alias
        # Topologically Sorted Source Nodes: [concated_features_32, concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_50.run(buf223, buf304, buf324, buf345, buf367, 2048, grid=grid(2048), stream=stream0)
        buf308 = reinterpret_tensor(buf309, (4, 32, 4, 4), (11264, 16, 4, 1), 10752)  # alias
        buf328 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 10752)  # alias
        buf349 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 10752)  # alias
        buf371 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 10752)  # alias
        # Topologically Sorted Source Nodes: [concated_features_32, concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_50.run(buf293, buf308, buf328, buf349, buf371, 2048, grid=grid(2048), stream=stream0)
        buf310 = empty_strided_cuda((4, 704, 4, 4), (11264, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_67, relu_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf309, primals_337, primals_338, primals_339, primals_340, buf310, 45056, grid=grid(45056), stream=stream0)
        del primals_340
        # Topologically Sorted Source Nodes: [bottleneck_output_32], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf312 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_68, relu_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf311, primals_342, primals_343, primals_344, primals_345, buf312, 8192, grid=grid(8192), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [new_features_32], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_346, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 32, 4, 4), (512, 16, 4, 1))
        buf325 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 9216)  # alias
        buf346 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 9216)  # alias
        buf368 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 9216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf239, buf325, buf346, buf368, 2048, grid=grid(2048), stream=stream0)
        buf329 = reinterpret_tensor(buf330, (4, 32, 4, 4), (11776, 16, 4, 1), 11264)  # alias
        buf350 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 11264)  # alias
        buf372 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 11264)  # alias
        # Topologically Sorted Source Nodes: [concated_features_33, concated_features_34, concated_features_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf313, buf329, buf350, buf372, 2048, grid=grid(2048), stream=stream0)
        buf331 = empty_strided_cuda((4, 736, 4, 4), (11776, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_69, relu_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf330, primals_347, primals_348, primals_349, primals_350, buf331, 47104, grid=grid(47104), stream=stream0)
        del primals_350
        # Topologically Sorted Source Nodes: [bottleneck_output_33], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf333 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_70, relu_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf332, primals_352, primals_353, primals_354, primals_355, buf333, 8192, grid=grid(8192), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [new_features_33], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 32, 4, 4), (512, 16, 4, 1))
        buf347 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 9728)  # alias
        buf369 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 9728)  # alias
        buf392 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 9728)  # alias
        # Topologically Sorted Source Nodes: [concated_features_34, concated_features_35, concated_features_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf256, buf347, buf369, buf392, 2048, grid=grid(2048), stream=stream0)
        buf351 = reinterpret_tensor(buf352, (4, 32, 4, 4), (12288, 16, 4, 1), 11776)  # alias
        buf373 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 11776)  # alias
        buf396 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 11776)  # alias
        # Topologically Sorted Source Nodes: [concated_features_34, concated_features_35, concated_features_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf334, buf351, buf373, buf396, 2048, grid=grid(2048), stream=stream0)
        buf353 = empty_strided_cuda((4, 768, 4, 4), (12288, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_71, relu_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf352, primals_357, primals_358, primals_359, primals_360, buf353, 49152, grid=grid(49152), stream=stream0)
        del primals_360
        # Topologically Sorted Source Nodes: [bottleneck_output_34], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf355 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_72, relu_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf354, primals_362, primals_363, primals_364, primals_365, buf355, 8192, grid=grid(8192), stream=stream0)
        del primals_365
        # Topologically Sorted Source Nodes: [new_features_34], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_366, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 32, 4, 4), (512, 16, 4, 1))
        buf358 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 4096)  # alias
        buf381 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 4096)  # alias
        buf405 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf159, buf358, buf381, buf405, 2048, grid=grid(2048), stream=stream0)
        buf359 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 4608)  # alias
        buf382 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 4608)  # alias
        buf406 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 4608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf164, buf359, buf382, buf406, 2048, grid=grid(2048), stream=stream0)
        buf360 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 5120)  # alias
        buf383 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 5120)  # alias
        buf407 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf169, buf360, buf383, buf407, 2048, grid=grid(2048), stream=stream0)
        buf361 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 5632)  # alias
        buf384 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 5632)  # alias
        buf408 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 5632)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf174, buf361, buf384, buf408, 2048, grid=grid(2048), stream=stream0)
        buf362 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 6144)  # alias
        buf385 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 6144)  # alias
        buf409 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf179, buf362, buf385, buf409, 2048, grid=grid(2048), stream=stream0)
        buf363 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 6656)  # alias
        buf386 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 6656)  # alias
        buf410 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 6656)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf184, buf363, buf386, buf410, 2048, grid=grid(2048), stream=stream0)
        buf364 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 7168)  # alias
        buf387 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 7168)  # alias
        buf411 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 7168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf189, buf364, buf387, buf411, 2048, grid=grid(2048), stream=stream0)
        buf365 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 7680)  # alias
        buf388 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 7680)  # alias
        buf412 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 7680)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf194, buf365, buf388, buf412, 2048, grid=grid(2048), stream=stream0)
        buf366 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 8192)  # alias
        buf389 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 8192)  # alias
        buf413 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf208, buf366, buf389, buf413, 2048, grid=grid(2048), stream=stream0)
        buf370 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 10240)  # alias
        buf393 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 10240)  # alias
        buf417 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf274, buf370, buf393, buf417, 2048, grid=grid(2048), stream=stream0)
        buf374 = reinterpret_tensor(buf375, (4, 32, 4, 4), (12800, 16, 4, 1), 12288)  # alias
        buf397 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 12288)  # alias
        buf421 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [concated_features_35, concated_features_36, concated_features_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_56.run(buf356, buf374, buf397, buf421, 2048, grid=grid(2048), stream=stream0)
        buf376 = empty_strided_cuda((4, 800, 4, 4), (12800, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_73, relu_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf375, primals_367, primals_368, primals_369, primals_370, buf376, 51200, grid=grid(51200), stream=stream0)
        del primals_370
        # Topologically Sorted Source Nodes: [bottleneck_output_35], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf378 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_74, relu_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf377, primals_372, primals_373, primals_374, primals_375, buf378, 8192, grid=grid(8192), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [new_features_35], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 32, 4, 4), (512, 16, 4, 1))
        buf390 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 8704)  # alias
        buf414 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 8704)  # alias
        buf439 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 8704)  # alias
        # Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf223, buf390, buf414, buf439, 2048, grid=grid(2048), stream=stream0)
        buf391 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 9216)  # alias
        buf415 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 9216)  # alias
        buf440 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 9216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf239, buf391, buf415, buf440, 2048, grid=grid(2048), stream=stream0)
        buf394 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 10752)  # alias
        buf418 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 10752)  # alias
        buf443 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 10752)  # alias
        # Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf293, buf394, buf418, buf443, 2048, grid=grid(2048), stream=stream0)
        buf395 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 11264)  # alias
        buf419 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 11264)  # alias
        buf444 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 11264)  # alias
        # Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf313, buf395, buf419, buf444, 2048, grid=grid(2048), stream=stream0)
        buf398 = reinterpret_tensor(buf399, (4, 32, 4, 4), (13312, 16, 4, 1), 12800)  # alias
        buf422 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 12800)  # alias
        buf447 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 12800)  # alias
        # Topologically Sorted Source Nodes: [concated_features_36, concated_features_37, concated_features_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf379, buf398, buf422, buf447, 2048, grid=grid(2048), stream=stream0)
        buf400 = empty_strided_cuda((4, 832, 4, 4), (13312, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_75, relu_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_59.run(buf399, primals_377, primals_378, primals_379, primals_380, buf400, 53248, grid=grid(53248), stream=stream0)
        del primals_380
        # Topologically Sorted Source Nodes: [bottleneck_output_36], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_381, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf402 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_76, relu_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf401, primals_382, primals_383, primals_384, primals_385, buf402, 8192, grid=grid(8192), stream=stream0)
        del primals_385
        # Topologically Sorted Source Nodes: [new_features_36], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_386, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 32, 4, 4), (512, 16, 4, 1))
        buf416 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 9728)  # alias
        buf441 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 9728)  # alias
        buf467 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 9728)  # alias
        # Topologically Sorted Source Nodes: [concated_features_37, concated_features_38, concated_features_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_60.run(buf256, buf416, buf441, buf467, 2048, grid=grid(2048), stream=stream0)
        buf420 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 11776)  # alias
        buf445 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 11776)  # alias
        buf471 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 11776)  # alias
        # Topologically Sorted Source Nodes: [concated_features_37, concated_features_38, concated_features_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_60.run(buf334, buf420, buf445, buf471, 2048, grid=grid(2048), stream=stream0)
        buf423 = reinterpret_tensor(buf424, (4, 32, 4, 4), (13824, 16, 4, 1), 13312)  # alias
        buf448 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 13312)  # alias
        buf474 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 13312)  # alias
        # Topologically Sorted Source Nodes: [concated_features_37, concated_features_38, concated_features_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_60.run(buf403, buf423, buf448, buf474, 2048, grid=grid(2048), stream=stream0)
        buf425 = empty_strided_cuda((4, 864, 4, 4), (13824, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_77, relu_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf424, primals_387, primals_388, primals_389, primals_390, buf425, 55296, grid=grid(55296), stream=stream0)
        del primals_390
        # Topologically Sorted Source Nodes: [bottleneck_output_37], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf427 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_78, relu_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf426, primals_392, primals_393, primals_394, primals_395, buf427, 8192, grid=grid(8192), stream=stream0)
        del primals_395
        # Topologically Sorted Source Nodes: [new_features_37], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 32, 4, 4), (512, 16, 4, 1))
        buf430 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 4096)  # alias
        buf456 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 4096)  # alias
        buf483 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf159, buf430, buf456, buf483, 2048, grid=grid(2048), stream=stream0)
        buf431 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 4608)  # alias
        buf457 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 4608)  # alias
        buf484 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 4608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf164, buf431, buf457, buf484, 2048, grid=grid(2048), stream=stream0)
        buf432 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 5120)  # alias
        buf458 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 5120)  # alias
        buf485 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf169, buf432, buf458, buf485, 2048, grid=grid(2048), stream=stream0)
        buf433 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 5632)  # alias
        buf459 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 5632)  # alias
        buf486 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 5632)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf174, buf433, buf459, buf486, 2048, grid=grid(2048), stream=stream0)
        buf434 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 6144)  # alias
        buf460 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 6144)  # alias
        buf487 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf179, buf434, buf460, buf487, 2048, grid=grid(2048), stream=stream0)
        buf435 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 6656)  # alias
        buf461 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 6656)  # alias
        buf488 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 6656)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf184, buf435, buf461, buf488, 2048, grid=grid(2048), stream=stream0)
        buf436 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 7168)  # alias
        buf462 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 7168)  # alias
        buf489 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 7168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf189, buf436, buf462, buf489, 2048, grid=grid(2048), stream=stream0)
        buf437 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 7680)  # alias
        buf463 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 7680)  # alias
        buf490 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 7680)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf194, buf437, buf463, buf490, 2048, grid=grid(2048), stream=stream0)
        buf438 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 8192)  # alias
        buf464 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 8192)  # alias
        buf491 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf208, buf438, buf464, buf491, 2048, grid=grid(2048), stream=stream0)
        buf442 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 10240)  # alias
        buf468 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 10240)  # alias
        buf495 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf274, buf442, buf468, buf495, 2048, grid=grid(2048), stream=stream0)
        buf446 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 12288)  # alias
        buf472 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 12288)  # alias
        buf499 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf356, buf446, buf472, buf499, 2048, grid=grid(2048), stream=stream0)
        buf449 = reinterpret_tensor(buf450, (4, 32, 4, 4), (14336, 16, 4, 1), 13824)  # alias
        buf475 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 13824)  # alias
        buf502 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 13824)  # alias
        # Topologically Sorted Source Nodes: [concated_features_38, concated_features_39, concated_features_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_62.run(buf428, buf449, buf475, buf502, 2048, grid=grid(2048), stream=stream0)
        buf451 = empty_strided_cuda((4, 896, 4, 4), (14336, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_79, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_63.run(buf450, primals_397, primals_398, primals_399, primals_400, buf451, 57344, grid=grid(57344), stream=stream0)
        del primals_400
        # Topologically Sorted Source Nodes: [bottleneck_output_38], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf453 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_80, relu_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf452, primals_402, primals_403, primals_404, primals_405, buf453, 8192, grid=grid(8192), stream=stream0)
        del primals_405
        # Topologically Sorted Source Nodes: [new_features_38], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (4, 32, 4, 4), (512, 16, 4, 1))
        buf465 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 8704)  # alias
        buf492 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 8704)  # alias
        buf520 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 8704)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf223, buf465, buf492, buf520, 2048, grid=grid(2048), stream=stream0)
        buf466 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 9216)  # alias
        buf493 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 9216)  # alias
        buf521 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 9216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf239, buf466, buf493, buf521, 2048, grid=grid(2048), stream=stream0)
        buf469 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 10752)  # alias
        buf496 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 10752)  # alias
        buf524 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 10752)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf293, buf469, buf496, buf524, 2048, grid=grid(2048), stream=stream0)
        buf470 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 11264)  # alias
        buf497 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 11264)  # alias
        buf525 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 11264)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf313, buf470, buf497, buf525, 2048, grid=grid(2048), stream=stream0)
        buf473 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 12800)  # alias
        buf500 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 12800)  # alias
        buf528 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 12800)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf379, buf473, buf500, buf528, 2048, grid=grid(2048), stream=stream0)
        buf476 = reinterpret_tensor(buf477, (4, 32, 4, 4), (14848, 16, 4, 1), 14336)  # alias
        buf503 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 14336)  # alias
        buf531 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 14336)  # alias
        # Topologically Sorted Source Nodes: [concated_features_39, concated_features_40, concated_features_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf454, buf476, buf503, buf531, 2048, grid=grid(2048), stream=stream0)
        buf478 = empty_strided_cuda((4, 928, 4, 4), (14848, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_81, relu_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_65.run(buf477, primals_407, primals_408, primals_409, primals_410, buf478, 59392, grid=grid(59392), stream=stream0)
        del primals_410
        # Topologically Sorted Source Nodes: [bottleneck_output_39], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_411, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf480 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_82, relu_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf479, primals_412, primals_413, primals_414, primals_415, buf480, 8192, grid=grid(8192), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [new_features_39], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_416, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 32, 4, 4), (512, 16, 4, 1))
        buf494 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 9728)  # alias
        buf522 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 9728)  # alias
        buf551 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 9728)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf256, buf494, buf522, buf551, 2048, grid=grid(2048), stream=stream0)
        del buf256
        buf498 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 11776)  # alias
        buf526 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 11776)  # alias
        buf555 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 11776)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf334, buf498, buf526, buf555, 2048, grid=grid(2048), stream=stream0)
        del buf334
        buf501 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 13312)  # alias
        buf529 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 13312)  # alias
        buf558 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 13312)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf403, buf501, buf529, buf558, 2048, grid=grid(2048), stream=stream0)
        del buf403
        buf504 = reinterpret_tensor(buf505, (4, 32, 4, 4), (15360, 16, 4, 1), 14848)  # alias
        buf532 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 14848)  # alias
        buf561 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 14848)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf481, buf504, buf532, buf561, 2048, grid=grid(2048), stream=stream0)
        del buf481
        buf506 = empty_strided_cuda((4, 960, 4, 4), (15360, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_83, relu_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_67.run(buf505, primals_417, primals_418, primals_419, primals_420, buf506, 61440, grid=grid(61440), stream=stream0)
        del primals_420
        # Topologically Sorted Source Nodes: [bottleneck_output_40], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_421, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf508 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_84, relu_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf507, primals_422, primals_423, primals_424, primals_425, buf508, 8192, grid=grid(8192), stream=stream0)
        del primals_425
        # Topologically Sorted Source Nodes: [new_features_40], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, primals_426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (4, 32, 4, 4), (512, 16, 4, 1))
        buf511 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 4096)  # alias
        buf540 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf159, buf511, buf540, 2048, grid=grid(2048), stream=stream0)
        del buf159
        buf512 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 4608)  # alias
        buf541 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 4608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf164, buf512, buf541, 2048, grid=grid(2048), stream=stream0)
        del buf164
        buf513 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 5120)  # alias
        buf542 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf169, buf513, buf542, 2048, grid=grid(2048), stream=stream0)
        buf514 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 5632)  # alias
        buf543 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 5632)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf174, buf514, buf543, 2048, grid=grid(2048), stream=stream0)
        buf515 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 6144)  # alias
        buf544 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf179, buf515, buf544, 2048, grid=grid(2048), stream=stream0)
        buf516 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 6656)  # alias
        buf545 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 6656)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf184, buf516, buf545, 2048, grid=grid(2048), stream=stream0)
        buf517 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 7168)  # alias
        buf546 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 7168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf189, buf517, buf546, 2048, grid=grid(2048), stream=stream0)
        buf518 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 7680)  # alias
        buf547 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 7680)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf194, buf518, buf547, 2048, grid=grid(2048), stream=stream0)
        buf519 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 8192)  # alias
        buf548 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf208, buf519, buf548, 2048, grid=grid(2048), stream=stream0)
        buf523 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 10240)  # alias
        buf552 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf274, buf523, buf552, 2048, grid=grid(2048), stream=stream0)
        buf527 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 12288)  # alias
        buf556 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf356, buf527, buf556, 2048, grid=grid(2048), stream=stream0)
        buf530 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 13824)  # alias
        buf559 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 13824)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf428, buf530, buf559, 2048, grid=grid(2048), stream=stream0)
        buf533 = reinterpret_tensor(buf534, (4, 32, 4, 4), (15872, 16, 4, 1), 15360)  # alias
        buf562 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 15360)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_68.run(buf509, buf533, buf562, 2048, grid=grid(2048), stream=stream0)
        buf535 = empty_strided_cuda((4, 992, 4, 4), (15872, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_85, relu_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_69.run(buf534, primals_427, primals_428, primals_429, primals_430, buf535, 63488, grid=grid(63488), stream=stream0)
        del primals_430
        # Topologically Sorted Source Nodes: [bottleneck_output_41], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_431, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf537 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_86, relu_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf536, primals_432, primals_433, primals_434, primals_435, buf537, 8192, grid=grid(8192), stream=stream0)
        del primals_435
        # Topologically Sorted Source Nodes: [new_features_41], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 32, 4, 4), (512, 16, 4, 1))
        buf549 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 8704)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf223, buf549, 2048, grid=grid(2048), stream=stream0)
        buf550 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 9216)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf239, buf550, 2048, grid=grid(2048), stream=stream0)
        buf553 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 10752)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf293, buf553, 2048, grid=grid(2048), stream=stream0)
        buf554 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 11264)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf313, buf554, 2048, grid=grid(2048), stream=stream0)
        buf557 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 12800)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf379, buf557, 2048, grid=grid(2048), stream=stream0)
        buf560 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 14336)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf454, buf560, 2048, grid=grid(2048), stream=stream0)
        buf563 = reinterpret_tensor(buf564, (4, 32, 4, 4), (16384, 16, 4, 1), 15872)  # alias
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_70.run(buf538, buf563, 2048, grid=grid(2048), stream=stream0)
        buf565 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_71.run(buf564, primals_437, primals_438, primals_439, primals_440, buf565, 65536, grid=grid(65536), stream=stream0)
        del primals_440
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_441, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf567 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf568 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        buf616 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        buf607 = reinterpret_tensor(buf616, (4, 512, 2, 2), (3072, 4, 2, 1), 0)  # alias
        buf631 = empty_strided_cuda((4, 800, 2, 2), (3200, 4, 2, 1), torch.float32)
        buf621 = reinterpret_tensor(buf631, (4, 512, 2, 2), (3200, 4, 2, 1), 0)  # alias
        buf647 = empty_strided_cuda((4, 832, 2, 2), (3328, 4, 2, 1), torch.float32)
        buf636 = reinterpret_tensor(buf647, (4, 512, 2, 2), (3328, 4, 2, 1), 0)  # alias
        buf664 = empty_strided_cuda((4, 864, 2, 2), (3456, 4, 2, 1), torch.float32)
        buf652 = reinterpret_tensor(buf664, (4, 512, 2, 2), (3456, 4, 2, 1), 0)  # alias
        buf682 = empty_strided_cuda((4, 896, 2, 2), (3584, 4, 2, 1), torch.float32)
        buf669 = reinterpret_tensor(buf682, (4, 512, 2, 2), (3584, 4, 2, 1), 0)  # alias
        buf701 = empty_strided_cuda((4, 928, 2, 2), (3712, 4, 2, 1), torch.float32)
        buf687 = reinterpret_tensor(buf701, (4, 512, 2, 2), (3712, 4, 2, 1), 0)  # alias
        buf721 = empty_strided_cuda((4, 960, 2, 2), (3840, 4, 2, 1), torch.float32)
        buf706 = reinterpret_tensor(buf721, (4, 512, 2, 2), (3840, 4, 2, 1), 0)  # alias
        buf742 = empty_strided_cuda((4, 992, 2, 2), (3968, 4, 2, 1), torch.float32)
        buf726 = reinterpret_tensor(buf742, (4, 512, 2, 2), (3968, 4, 2, 1), 0)  # alias
        buf764 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        buf747 = reinterpret_tensor(buf764, (4, 512, 2, 2), (4096, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_19, batch_norm_88, relu_88, concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54, concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_72.run(buf566, primals_442, primals_443, primals_444, primals_445, buf567, buf568, buf607, buf621, buf636, buf652, buf669, buf687, buf706, buf726, buf747, 8192, grid=grid(8192), stream=stream0)
        del primals_445
        # Topologically Sorted Source Nodes: [bottleneck_output_42], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf568, primals_446, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (4, 128, 2, 2), (512, 4, 2, 1))
        buf570 = reinterpret_tensor(buf538, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_89, relu_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf569, primals_447, primals_448, primals_449, primals_450, buf570, 2048, grid=grid(2048), stream=stream0)
        del primals_450
        # Topologically Sorted Source Nodes: [new_features_42], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, primals_451, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (4, 32, 2, 2), (128, 4, 2, 1))
        buf572 = empty_strided_cuda((4, 544, 2, 2), (2176, 4, 2, 1), torch.float32)
        buf573 = empty_strided_cuda((4, 544, 2, 2), (2176, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_43, batch_norm_90, relu_90], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_74.run(buf567, buf571, primals_452, primals_453, primals_454, primals_455, buf572, buf573, 8704, grid=grid(8704), stream=stream0)
        del primals_455
        # Topologically Sorted Source Nodes: [bottleneck_output_43], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 128, 2, 2), (512, 4, 2, 1))
        buf575 = reinterpret_tensor(buf454, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_91, relu_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf574, primals_457, primals_458, primals_459, primals_460, buf575, 2048, grid=grid(2048), stream=stream0)
        del primals_460
        # Topologically Sorted Source Nodes: [new_features_43], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_461, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 32, 2, 2), (128, 4, 2, 1))
        buf577 = empty_strided_cuda((4, 576, 2, 2), (2304, 4, 2, 1), torch.float32)
        buf578 = empty_strided_cuda((4, 576, 2, 2), (2304, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_44, batch_norm_92, relu_92], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_75.run(buf567, buf571, buf576, primals_462, primals_463, primals_464, primals_465, buf577, buf578, 9216, grid=grid(9216), stream=stream0)
        del primals_465
        # Topologically Sorted Source Nodes: [bottleneck_output_44], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_466, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 128, 2, 2), (512, 4, 2, 1))
        buf580 = reinterpret_tensor(buf379, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_93, relu_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf579, primals_467, primals_468, primals_469, primals_470, buf580, 2048, grid=grid(2048), stream=stream0)
        del primals_470
        # Topologically Sorted Source Nodes: [new_features_44], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, primals_471, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (4, 32, 2, 2), (128, 4, 2, 1))
        buf582 = empty_strided_cuda((4, 608, 2, 2), (2432, 4, 2, 1), torch.float32)
        buf583 = empty_strided_cuda((4, 608, 2, 2), (2432, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_45, batch_norm_94, relu_94], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_76.run(buf567, buf571, buf576, buf581, primals_472, primals_473, primals_474, primals_475, buf582, buf583, 9728, grid=grid(9728), stream=stream0)
        del primals_475
        # Topologically Sorted Source Nodes: [bottleneck_output_45], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_476, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 128, 2, 2), (512, 4, 2, 1))
        buf585 = reinterpret_tensor(buf313, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_95, relu_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf584, primals_477, primals_478, primals_479, primals_480, buf585, 2048, grid=grid(2048), stream=stream0)
        del primals_480
        # Topologically Sorted Source Nodes: [new_features_45], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_481, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (4, 32, 2, 2), (128, 4, 2, 1))
        buf587 = empty_strided_cuda((4, 640, 2, 2), (2560, 4, 2, 1), torch.float32)
        buf588 = empty_strided_cuda((4, 640, 2, 2), (2560, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_46, batch_norm_96, relu_96], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_77.run(buf567, buf571, buf576, buf581, buf586, primals_482, primals_483, primals_484, primals_485, buf587, buf588, 10240, grid=grid(10240), stream=stream0)
        del primals_485
        # Topologically Sorted Source Nodes: [bottleneck_output_46], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (4, 128, 2, 2), (512, 4, 2, 1))
        buf590 = reinterpret_tensor(buf293, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_97, relu_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf589, primals_487, primals_488, primals_489, primals_490, buf590, 2048, grid=grid(2048), stream=stream0)
        del primals_490
        # Topologically Sorted Source Nodes: [new_features_46], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_491, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (4, 32, 2, 2), (128, 4, 2, 1))
        buf592 = empty_strided_cuda((4, 672, 2, 2), (2688, 4, 2, 1), torch.float32)
        buf593 = empty_strided_cuda((4, 672, 2, 2), (2688, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_47, batch_norm_98, relu_98], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_78.run(buf567, buf571, buf576, buf581, buf586, buf591, primals_492, primals_493, primals_494, primals_495, buf592, buf593, 10752, grid=grid(10752), stream=stream0)
        del primals_495
        # Topologically Sorted Source Nodes: [bottleneck_output_47], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf593, primals_496, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (4, 128, 2, 2), (512, 4, 2, 1))
        buf595 = reinterpret_tensor(buf239, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_99, relu_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf594, primals_497, primals_498, primals_499, primals_500, buf595, 2048, grid=grid(2048), stream=stream0)
        del primals_500
        # Topologically Sorted Source Nodes: [new_features_47], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_501, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 32, 2, 2), (128, 4, 2, 1))
        buf597 = empty_strided_cuda((4, 704, 2, 2), (2816, 4, 2, 1), torch.float32)
        buf598 = empty_strided_cuda((4, 704, 2, 2), (2816, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_48, batch_norm_100, relu_100], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79.run(buf567, buf571, buf576, buf581, buf586, buf591, buf596, primals_502, primals_503, primals_504, primals_505, buf597, buf598, 11264, grid=grid(11264), stream=stream0)
        del primals_505
        # Topologically Sorted Source Nodes: [bottleneck_output_48], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf598, primals_506, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (4, 128, 2, 2), (512, 4, 2, 1))
        buf600 = reinterpret_tensor(buf223, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_101, relu_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf599, primals_507, primals_508, primals_509, primals_510, buf600, 2048, grid=grid(2048), stream=stream0)
        del primals_510
        # Topologically Sorted Source Nodes: [new_features_48], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_511, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (4, 32, 2, 2), (128, 4, 2, 1))
        buf602 = empty_strided_cuda((4, 736, 2, 2), (2944, 4, 2, 1), torch.float32)
        buf603 = empty_strided_cuda((4, 736, 2, 2), (2944, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_49, batch_norm_102, relu_102], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_80.run(buf567, buf571, buf576, buf581, buf586, buf591, buf596, buf601, primals_512, primals_513, primals_514, primals_515, buf602, buf603, 11776, grid=grid(11776), stream=stream0)
        del primals_515
        # Topologically Sorted Source Nodes: [bottleneck_output_49], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(buf603, primals_516, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf604, (4, 128, 2, 2), (512, 4, 2, 1))
        buf605 = reinterpret_tensor(buf509, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_103, relu_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf604, primals_517, primals_518, primals_519, primals_520, buf605, 2048, grid=grid(2048), stream=stream0)
        del primals_520
        # Topologically Sorted Source Nodes: [new_features_49], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_521, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 32, 2, 2), (128, 4, 2, 1))
        buf608 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2048)  # alias
        buf622 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2048)  # alias
        buf637 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2048)  # alias
        buf653 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2048)  # alias
        buf670 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf571, buf608, buf622, buf637, buf653, buf670, 512, grid=grid(512), stream=stream0)
        buf609 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2176)  # alias
        buf623 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2176)  # alias
        buf638 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2176)  # alias
        buf654 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2176)  # alias
        buf671 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2176)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf576, buf609, buf623, buf638, buf654, buf671, 512, grid=grid(512), stream=stream0)
        buf610 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2304)  # alias
        buf624 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2304)  # alias
        buf639 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2304)  # alias
        buf655 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2304)  # alias
        buf672 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2304)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf581, buf610, buf624, buf639, buf655, buf672, 512, grid=grid(512), stream=stream0)
        buf611 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2432)  # alias
        buf625 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2432)  # alias
        buf640 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2432)  # alias
        buf656 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2432)  # alias
        buf673 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2432)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf586, buf611, buf625, buf640, buf656, buf673, 512, grid=grid(512), stream=stream0)
        buf612 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2560)  # alias
        buf626 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2560)  # alias
        buf641 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2560)  # alias
        buf657 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2560)  # alias
        buf674 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2560)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf591, buf612, buf626, buf641, buf657, buf674, 512, grid=grid(512), stream=stream0)
        buf613 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2688)  # alias
        buf627 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2688)  # alias
        buf642 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2688)  # alias
        buf658 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2688)  # alias
        buf675 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2688)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf596, buf613, buf627, buf642, buf658, buf675, 512, grid=grid(512), stream=stream0)
        buf614 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2816)  # alias
        buf628 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2816)  # alias
        buf643 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2816)  # alias
        buf659 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2816)  # alias
        buf676 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2816)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf601, buf614, buf628, buf643, buf659, buf676, 512, grid=grid(512), stream=stream0)
        buf615 = reinterpret_tensor(buf616, (4, 32, 2, 2), (3072, 4, 2, 1), 2944)  # alias
        buf629 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 2944)  # alias
        buf644 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 2944)  # alias
        buf660 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 2944)  # alias
        buf677 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 2944)  # alias
        # Topologically Sorted Source Nodes: [concated_features_50, concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf606, buf615, buf629, buf644, buf660, buf677, 512, grid=grid(512), stream=stream0)
        buf617 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_104, relu_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_82.run(buf616, primals_522, primals_523, primals_524, primals_525, buf617, 12288, grid=grid(12288), stream=stream0)
        del primals_525
        # Topologically Sorted Source Nodes: [bottleneck_output_50], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_526, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (4, 128, 2, 2), (512, 4, 2, 1))
        buf619 = reinterpret_tensor(buf428, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_105, relu_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf618, primals_527, primals_528, primals_529, primals_530, buf619, 2048, grid=grid(2048), stream=stream0)
        del primals_530
        # Topologically Sorted Source Nodes: [new_features_50], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_531, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 32, 2, 2), (128, 4, 2, 1))
        buf630 = reinterpret_tensor(buf631, (4, 32, 2, 2), (3200, 4, 2, 1), 3072)  # alias
        buf645 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 3072)  # alias
        buf661 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 3072)  # alias
        buf678 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 3072)  # alias
        # Topologically Sorted Source Nodes: [concated_features_51, concated_features_52, concated_features_53, concated_features_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_83.run(buf620, buf630, buf645, buf661, buf678, 512, grid=grid(512), stream=stream0)
        buf632 = empty_strided_cuda((4, 800, 2, 2), (3200, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_106, relu_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf631, primals_532, primals_533, primals_534, primals_535, buf632, 12800, grid=grid(12800), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [bottleneck_output_51], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (4, 128, 2, 2), (512, 4, 2, 1))
        buf634 = reinterpret_tensor(buf356, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_107, relu_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf633, primals_537, primals_538, primals_539, primals_540, buf634, 2048, grid=grid(2048), stream=stream0)
        del primals_540
        # Topologically Sorted Source Nodes: [new_features_51], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf634, primals_541, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (4, 32, 2, 2), (128, 4, 2, 1))
        buf646 = reinterpret_tensor(buf647, (4, 32, 2, 2), (3328, 4, 2, 1), 3200)  # alias
        buf662 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 3200)  # alias
        buf679 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 3200)  # alias
        buf697 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 3200)  # alias
        # Topologically Sorted Source Nodes: [concated_features_52, concated_features_53, concated_features_54, concated_features_55], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_85.run(buf635, buf646, buf662, buf679, buf697, 512, grid=grid(512), stream=stream0)
        buf648 = empty_strided_cuda((4, 832, 2, 2), (3328, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_108, relu_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_86.run(buf647, primals_542, primals_543, primals_544, primals_545, buf648, 13312, grid=grid(13312), stream=stream0)
        del primals_545
        # Topologically Sorted Source Nodes: [bottleneck_output_52], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (4, 128, 2, 2), (512, 4, 2, 1))
        buf650 = reinterpret_tensor(buf274, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_109, relu_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf649, primals_547, primals_548, primals_549, primals_550, buf650, 2048, grid=grid(2048), stream=stream0)
        del primals_550
        # Topologically Sorted Source Nodes: [new_features_52], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, primals_551, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (4, 32, 2, 2), (128, 4, 2, 1))
        buf663 = reinterpret_tensor(buf664, (4, 32, 2, 2), (3456, 4, 2, 1), 3328)  # alias
        buf680 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 3328)  # alias
        buf698 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 3328)  # alias
        buf717 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3328)  # alias
        # Topologically Sorted Source Nodes: [concated_features_53, concated_features_54, concated_features_55, concated_features_56], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_87.run(buf651, buf663, buf680, buf698, buf717, 512, grid=grid(512), stream=stream0)
        buf665 = empty_strided_cuda((4, 864, 2, 2), (3456, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_110, relu_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_88.run(buf664, primals_552, primals_553, primals_554, primals_555, buf665, 13824, grid=grid(13824), stream=stream0)
        del primals_555
        # Topologically Sorted Source Nodes: [bottleneck_output_53], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (4, 128, 2, 2), (512, 4, 2, 1))
        buf667 = reinterpret_tensor(buf208, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_111, relu_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf666, primals_557, primals_558, primals_559, primals_560, buf667, 2048, grid=grid(2048), stream=stream0)
        del primals_560
        # Topologically Sorted Source Nodes: [new_features_53], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf667, primals_561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (4, 32, 2, 2), (128, 4, 2, 1))
        buf681 = reinterpret_tensor(buf682, (4, 32, 2, 2), (3584, 4, 2, 1), 3456)  # alias
        buf699 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 3456)  # alias
        buf718 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3456)  # alias
        buf738 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3456)  # alias
        # Topologically Sorted Source Nodes: [concated_features_54, concated_features_55, concated_features_56, concated_features_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_89.run(buf668, buf681, buf699, buf718, buf738, 512, grid=grid(512), stream=stream0)
        buf683 = empty_strided_cuda((4, 896, 2, 2), (3584, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_112, relu_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_90.run(buf682, primals_562, primals_563, primals_564, primals_565, buf683, 14336, grid=grid(14336), stream=stream0)
        del primals_565
        # Topologically Sorted Source Nodes: [bottleneck_output_54], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_566, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (4, 128, 2, 2), (512, 4, 2, 1))
        buf685 = reinterpret_tensor(buf194, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_113, relu_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf684, primals_567, primals_568, primals_569, primals_570, buf685, 2048, grid=grid(2048), stream=stream0)
        del primals_570
        # Topologically Sorted Source Nodes: [new_features_54], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, primals_571, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (4, 32, 2, 2), (128, 4, 2, 1))
        buf688 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2048)  # alias
        buf707 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2048)  # alias
        buf727 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2048)  # alias
        buf748 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf571, buf688, buf707, buf727, buf748, 512, grid=grid(512), stream=stream0)
        del buf571
        buf689 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2176)  # alias
        buf708 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2176)  # alias
        buf728 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2176)  # alias
        buf749 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2176)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf576, buf689, buf708, buf728, buf749, 512, grid=grid(512), stream=stream0)
        del buf576
        buf690 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2304)  # alias
        buf709 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2304)  # alias
        buf729 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2304)  # alias
        buf750 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2304)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf581, buf690, buf709, buf729, buf750, 512, grid=grid(512), stream=stream0)
        del buf581
        buf691 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2432)  # alias
        buf710 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2432)  # alias
        buf730 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2432)  # alias
        buf751 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2432)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf586, buf691, buf710, buf730, buf751, 512, grid=grid(512), stream=stream0)
        del buf586
        buf692 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2560)  # alias
        buf711 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2560)  # alias
        buf731 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2560)  # alias
        buf752 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2560)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf591, buf692, buf711, buf731, buf752, 512, grid=grid(512), stream=stream0)
        del buf591
        buf693 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2688)  # alias
        buf712 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2688)  # alias
        buf732 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2688)  # alias
        buf753 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2688)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf596, buf693, buf712, buf732, buf753, 512, grid=grid(512), stream=stream0)
        del buf596
        buf694 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2816)  # alias
        buf713 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2816)  # alias
        buf733 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2816)  # alias
        buf754 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2816)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf601, buf694, buf713, buf733, buf754, 512, grid=grid(512), stream=stream0)
        del buf601
        buf695 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 2944)  # alias
        buf714 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 2944)  # alias
        buf734 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 2944)  # alias
        buf755 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 2944)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf606, buf695, buf714, buf734, buf755, 512, grid=grid(512), stream=stream0)
        del buf606
        buf696 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 3072)  # alias
        buf715 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3072)  # alias
        buf735 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3072)  # alias
        buf756 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3072)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf620, buf696, buf715, buf735, buf756, 512, grid=grid(512), stream=stream0)
        del buf620
        buf700 = reinterpret_tensor(buf701, (4, 32, 2, 2), (3712, 4, 2, 1), 3584)  # alias
        buf719 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3584)  # alias
        buf739 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3584)  # alias
        buf760 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3584)  # alias
        # Topologically Sorted Source Nodes: [concated_features_55, concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_91.run(buf686, buf700, buf719, buf739, buf760, 512, grid=grid(512), stream=stream0)
        del buf686
        buf702 = empty_strided_cuda((4, 928, 2, 2), (3712, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_114, relu_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_92.run(buf701, primals_572, primals_573, primals_574, primals_575, buf702, 14848, grid=grid(14848), stream=stream0)
        del primals_575
        # Topologically Sorted Source Nodes: [bottleneck_output_55], Original ATen: [aten.convolution]
        buf703 = extern_kernels.convolution(buf702, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (4, 128, 2, 2), (512, 4, 2, 1))
        buf704 = reinterpret_tensor(buf189, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_115, relu_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf703, primals_577, primals_578, primals_579, primals_580, buf704, 2048, grid=grid(2048), stream=stream0)
        del primals_580
        # Topologically Sorted Source Nodes: [new_features_55], Original ATen: [aten.convolution]
        buf705 = extern_kernels.convolution(buf704, primals_581, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (4, 32, 2, 2), (128, 4, 2, 1))
        buf716 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3200)  # alias
        buf736 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3200)  # alias
        buf757 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3200)  # alias
        # Topologically Sorted Source Nodes: [concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_93.run(buf635, buf716, buf736, buf757, 512, grid=grid(512), stream=stream0)
        del buf635
        buf720 = reinterpret_tensor(buf721, (4, 32, 2, 2), (3840, 4, 2, 1), 3712)  # alias
        buf740 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3712)  # alias
        buf761 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3712)  # alias
        # Topologically Sorted Source Nodes: [concated_features_56, concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_93.run(buf705, buf720, buf740, buf761, 512, grid=grid(512), stream=stream0)
        del buf705
        buf722 = empty_strided_cuda((4, 960, 2, 2), (3840, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_116, relu_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_94.run(buf721, primals_582, primals_583, primals_584, primals_585, buf722, 15360, grid=grid(15360), stream=stream0)
        del primals_585
        # Topologically Sorted Source Nodes: [bottleneck_output_56], Original ATen: [aten.convolution]
        buf723 = extern_kernels.convolution(buf722, primals_586, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf723, (4, 128, 2, 2), (512, 4, 2, 1))
        buf724 = reinterpret_tensor(buf184, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_117, relu_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf723, primals_587, primals_588, primals_589, primals_590, buf724, 2048, grid=grid(2048), stream=stream0)
        del primals_590
        # Topologically Sorted Source Nodes: [new_features_56], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_591, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf725, (4, 32, 2, 2), (128, 4, 2, 1))
        buf737 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3328)  # alias
        buf758 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3328)  # alias
        # Topologically Sorted Source Nodes: [concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_95.run(buf651, buf737, buf758, 512, grid=grid(512), stream=stream0)
        del buf651
        buf741 = reinterpret_tensor(buf742, (4, 32, 2, 2), (3968, 4, 2, 1), 3840)  # alias
        buf762 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3840)  # alias
        # Topologically Sorted Source Nodes: [concated_features_57, input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_95.run(buf725, buf741, buf762, 512, grid=grid(512), stream=stream0)
        del buf725
        buf743 = empty_strided_cuda((4, 992, 2, 2), (3968, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_118, relu_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_96.run(buf742, primals_592, primals_593, primals_594, primals_595, buf743, 15872, grid=grid(15872), stream=stream0)
        del primals_595
        # Topologically Sorted Source Nodes: [bottleneck_output_57], Original ATen: [aten.convolution]
        buf744 = extern_kernels.convolution(buf743, primals_596, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf744, (4, 128, 2, 2), (512, 4, 2, 1))
        buf745 = reinterpret_tensor(buf179, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_119, relu_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_73.run(buf744, primals_597, primals_598, primals_599, primals_600, buf745, 2048, grid=grid(2048), stream=stream0)
        del primals_600
        # Topologically Sorted Source Nodes: [new_features_57], Original ATen: [aten.convolution]
        buf746 = extern_kernels.convolution(buf745, primals_601, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf746, (4, 32, 2, 2), (128, 4, 2, 1))
        buf759 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3456)  # alias
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_97.run(buf668, buf759, 512, grid=grid(512), stream=stream0)
        del buf668
        buf763 = reinterpret_tensor(buf764, (4, 32, 2, 2), (4096, 4, 2, 1), 3968)  # alias
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_97.run(buf746, buf763, 512, grid=grid(512), stream=stream0)
        del buf746
        buf765 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_98.run(buf764, primals_602, primals_603, primals_604, primals_605, buf765, 4096, grid=grid(4096), stream=stream0)
        del primals_605
        buf766 = reinterpret_tensor(buf174, (4, 512), (512, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_607, reinterpret_tensor(buf765, (4, 1024), (1024, 1), 0), reinterpret_tensor(primals_606, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf766)
        del primals_607
        buf767 = reinterpret_tensor(buf169, (4, 512), (512, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_99.run(buf766, primals_608, primals_609, primals_610, primals_611, buf767, 2048, grid=grid(2048), stream=stream0)
        del primals_611
        buf768 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_613, buf767, reinterpret_tensor(primals_612, (512, 4), (1, 512), 0), alpha=1, beta=1, out=buf768)
        del primals_613
    return (buf768, primals_1, primals_2, primals_3, primals_4, primals_5, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_89, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, primals_169, primals_171, primals_172, primals_173, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_182, primals_183, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_202, primals_203, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_212, primals_213, primals_214, primals_216, primals_217, primals_218, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_254, primals_256, primals_257, primals_258, primals_259, primals_261, primals_262, primals_263, primals_264, primals_266, primals_267, primals_268, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_277, primals_278, primals_279, primals_281, primals_282, primals_283, primals_284, primals_286, primals_287, primals_288, primals_289, primals_291, primals_292, primals_293, primals_294, primals_296, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_326, primals_327, primals_328, primals_329, primals_331, primals_332, primals_333, primals_334, primals_336, primals_337, primals_338, primals_339, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_351, primals_352, primals_353, primals_354, primals_356, primals_357, primals_358, primals_359, primals_361, primals_362, primals_363, primals_364, primals_366, primals_367, primals_368, primals_369, primals_371, primals_372, primals_373, primals_374, primals_376, primals_377, primals_378, primals_379, primals_381, primals_382, primals_383, primals_384, primals_386, primals_387, primals_388, primals_389, primals_391, primals_392, primals_393, primals_394, primals_396, primals_397, primals_398, primals_399, primals_401, primals_402, primals_403, primals_404, primals_406, primals_407, primals_408, primals_409, primals_411, primals_412, primals_413, primals_414, primals_416, primals_417, primals_418, primals_419, primals_421, primals_422, primals_423, primals_424, primals_426, primals_427, primals_428, primals_429, primals_431, primals_432, primals_433, primals_434, primals_436, primals_437, primals_438, primals_439, primals_441, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_449, primals_451, primals_452, primals_453, primals_454, primals_456, primals_457, primals_458, primals_459, primals_461, primals_462, primals_463, primals_464, primals_466, primals_467, primals_468, primals_469, primals_471, primals_472, primals_473, primals_474, primals_476, primals_477, primals_478, primals_479, primals_481, primals_482, primals_483, primals_484, primals_486, primals_487, primals_488, primals_489, primals_491, primals_492, primals_493, primals_494, primals_496, primals_497, primals_498, primals_499, primals_501, primals_502, primals_503, primals_504, primals_506, primals_507, primals_508, primals_509, primals_511, primals_512, primals_513, primals_514, primals_516, primals_517, primals_518, primals_519, primals_521, primals_522, primals_523, primals_524, primals_526, primals_527, primals_528, primals_529, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_541, primals_542, primals_543, primals_544, primals_546, primals_547, primals_548, primals_549, primals_551, primals_552, primals_553, primals_554, primals_556, primals_557, primals_558, primals_559, primals_561, primals_562, primals_563, primals_564, primals_566, primals_567, primals_568, primals_569, primals_571, primals_572, primals_573, primals_574, primals_576, primals_577, primals_578, primals_579, primals_581, primals_582, primals_583, primals_584, primals_586, primals_587, primals_588, primals_589, primals_591, primals_592, primals_593, primals_594, primals_596, primals_597, primals_598, primals_599, primals_601, primals_602, primals_603, primals_604, primals_608, primals_609, primals_610, buf0, buf1, buf3, buf4, buf5, buf6, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf24, buf25, buf26, buf27, buf29, buf30, buf31, buf32, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf42, buf43, buf44, buf45, buf47, buf48, buf49, buf50, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf62, buf63, buf64, buf65, buf67, buf68, buf69, buf70, buf72, buf73, buf74, buf75, buf86, buf87, buf88, buf89, buf101, buf102, buf103, buf104, buf117, buf118, buf119, buf120, buf134, buf135, buf136, buf137, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf160, buf161, buf162, buf163, buf165, buf166, buf167, buf168, buf170, buf171, buf172, buf173, buf175, buf176, buf177, buf178, buf180, buf181, buf182, buf183, buf185, buf186, buf187, buf188, buf190, buf191, buf192, buf193, buf204, buf205, buf206, buf207, buf219, buf220, buf221, buf222, buf235, buf236, buf237, buf238, buf252, buf253, buf254, buf255, buf270, buf271, buf272, buf273, buf289, buf290, buf291, buf292, buf309, buf310, buf311, buf312, buf330, buf331, buf332, buf333, buf352, buf353, buf354, buf355, buf375, buf376, buf377, buf378, buf399, buf400, buf401, buf402, buf424, buf425, buf426, buf427, buf450, buf451, buf452, buf453, buf477, buf478, buf479, buf480, buf505, buf506, buf507, buf508, buf534, buf535, buf536, buf537, buf564, buf565, buf566, buf567, buf568, buf569, buf570, buf572, buf573, buf574, buf575, buf577, buf578, buf579, buf580, buf582, buf583, buf584, buf585, buf587, buf588, buf589, buf590, buf592, buf593, buf594, buf595, buf597, buf598, buf599, buf600, buf602, buf603, buf604, buf605, buf616, buf617, buf618, buf619, buf631, buf632, buf633, buf634, buf647, buf648, buf649, buf650, buf664, buf665, buf666, buf667, buf682, buf683, buf684, buf685, buf701, buf702, buf703, buf704, buf721, buf722, buf723, buf724, buf742, buf743, buf744, buf745, buf764, reinterpret_tensor(buf765, (4, 1024), (1024, 1), 0), buf766, buf767, primals_612, primals_606, buf769, )


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
    primals_11 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
