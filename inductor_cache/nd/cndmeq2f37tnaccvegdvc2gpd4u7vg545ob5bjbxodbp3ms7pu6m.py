# AOT ID: ['22_forward']
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


# kernel path: inductor_cache/ko/ckoidgcxmfqmhvevmbelljzdwj3e3nm2osp7fw7hcc3wlqsfqj26.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_4 => getitem, getitem_1
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex // 16
    x4 = xindex
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
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x3), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x3), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x3), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x3), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x3), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x3), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x3), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x3), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x3), tmp49, eviction_policy='evict_last', other=float("-inf"))
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


# kernel path: inductor_cache/45/c45te3gf6p2uyvfymkpk7k62e32m3hvl55qijfda7nrtq2ax5qyt.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_3, mul_4, mul_5, sub_1
#   out_2 => relu_1
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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


# kernel path: inductor_cache/cb/ccbfhut7rwruxjhihxspeqar7bumhnwyytlhsuvftnut24c6tgd4.py
# Topologically Sorted Source Nodes: [out_7, input_6, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_9, mul_13, mul_14, sub_4
#   out_7 => add_7, mul_10, mul_11, sub_3
#   out_8 => add_10
#   out_9 => relu_3
# Graph fragment:
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
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/ky/ckyxdgzf6svclfgc53eoeiihbcun5lhxwkrukef4bysm6h4bunjo.py
# Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_17 => add_16, mul_22, mul_23, sub_7
#   out_18 => add_17
#   out_19 => relu_6
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %relu_3), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
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


# kernel path: inductor_cache/gx/cgxujiclmwou6q7b66gmos6qkqgderrvj36gceac4qyhc273fu3p.py
# Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_31 => add_26, mul_34, mul_35, sub_11
#   out_32 => relu_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ye/cyen5uv3a5hj4qyf52etq5dbly5i6nwwobax7omfso2wlftgynnw.py
# Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_34 => add_28, mul_37, mul_38, sub_12
#   out_35 => relu_11
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3u/c3uy2f42zo35io4jwc65xzbpkat4bvqopsraotexkz4vcha6pd2j.py
# Topologically Sorted Source Nodes: [out_37, input_8, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_32, mul_43, mul_44, sub_14
#   out_37 => add_30, mul_40, mul_41, sub_13
#   out_38 => add_33
#   out_39 => relu_12
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_32), kwargs = {})
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqs6ewqjv5wiismk5hodmwhjonm2or7kihod7vfkwuidzehaiues.py
# Topologically Sorted Source Nodes: [out_47, out_48, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_47 => add_39, mul_52, mul_53, sub_17
#   out_48 => add_40
#   out_49 => relu_15
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %relu_12), kwargs = {})
#   %relu_15 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_40,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/r3/cr33cmkdqg2rv6m2grukctgbljw32eyrtep4beiv7gmlrsn3mvqn.py
# Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_111 => add_84, mul_109, mul_110, sub_36
#   out_112 => relu_34
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %relu_34 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_84,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
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


# kernel path: inductor_cache/a3/ca3dgfi5nnjll374ct3biii75zjo6ekhz3dhbnjegfoz6tftqbki.py
# Topologically Sorted Source Nodes: [out_117, input_10, out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_90, mul_118, mul_119, sub_39
#   out_117 => add_88, mul_115, mul_116, sub_38
#   out_118 => add_91
#   out_119 => relu_36
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %add_90), kwargs = {})
#   %relu_36 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/f2/cf27d4tgn5xjazky467jppkey26p5x5rg3trfqakkqt3yv254mxv.py
# Topologically Sorted Source Nodes: [out_127, out_128, out_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_127 => add_97, mul_127, mul_128, sub_42
#   out_128 => add_98
#   out_129 => relu_39
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_97, %relu_36), kwargs = {})
#   %relu_39 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_98,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 1024)
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


# kernel path: inductor_cache/oo/coogff2lumhgrsc5mcte5mdcz3ligedvdlqdreakte7nl2hppfwy.py
# Topologically Sorted Source Nodes: [out_471, out_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_471 => add_338, mul_436, mul_437, sub_145
#   out_472 => relu_142
# Graph fragment:
#   %sub_145 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_1161), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_145, %unsqueeze_1163), kwargs = {})
#   %mul_437 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_436, %unsqueeze_1165), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_437, %unsqueeze_1167), kwargs = {})
#   %relu_142 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_338,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/px/cpxt2gfthtrcfol5pbab6yuayllizyi5375w6uagditcy5pddgfa.py
# Topologically Sorted Source Nodes: [out_477, input_12, out_478, out_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_344, mul_445, mul_446, sub_148
#   out_477 => add_342, mul_442, mul_443, sub_147
#   out_478 => add_345
#   out_479 => relu_144
# Graph fragment:
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_147, %unsqueeze_1177), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1179), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %unsqueeze_1181), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_443, %unsqueeze_1183), kwargs = {})
#   %sub_148 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_148, %unsqueeze_1185), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_148, %unsqueeze_1187), kwargs = {})
#   %mul_446 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, %unsqueeze_1189), kwargs = {})
#   %add_344 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_446, %unsqueeze_1191), kwargs = {})
#   %add_345 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_342, %add_344), kwargs = {})
#   %relu_144 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_345,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqxktkszlj5sr6iaolpkcpkmegsln5f6efnki4zqf2ygbfejpax.py
# Topologically Sorted Source Nodes: [out_487, out_488, out_489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_487 => add_351, mul_454, mul_455, sub_151
#   out_488 => add_352
#   out_489 => relu_147
# Graph fragment:
#   %sub_151 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_151, %unsqueeze_1209), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_151, %unsqueeze_1211), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_454, %unsqueeze_1213), kwargs = {})
#   %add_351 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %unsqueeze_1215), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_351, %relu_144), kwargs = {})
#   %relu_147 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_352,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 2048)
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


# kernel path: inductor_cache/kn/cknvpzdhy6ax6po33e2rgy4zd64sdm4ceejaui4gmh36hmtp6gp4.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution_155
# Graph fragment:
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_150, %primals_777, %primals_778, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3cdbhjxmsend7ypfcp7e33aht3v6kggiwqjtrzpzqupjxvjtpwi.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pixel_shuffle]
# Source node to ATen node mapping:
#   x_3 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_pixel_shuffle_16 = async_compile.triton('triton_poi_fused_pixel_shuffle_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 8}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pixel_shuffle_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pixel_shuffle_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y0 = (yindex % 8)
    y1 = ((yindex // 8) % 8)
    y2 = ((yindex // 64) % 8)
    y6 = yindex // 512
    y3 = ((yindex // 512) % 4)
    y7 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8*y2 + 64*x5 + 512*y1 + 4096*y6), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + 8*y1 + 64*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + 8*y1 + 64*y3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x5 + 8*y1 + 64*y3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x5 + 8*y1 + 64*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
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
    tl.store(out_ptr0 + (x5 + 8*y7), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (512, ), (1, ))
    assert_size_stride(primals_137, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_186, (256, ), (1, ))
    assert_size_stride(primals_187, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_193, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_198, (1024, ), (1, ))
    assert_size_stride(primals_199, (1024, ), (1, ))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, ), (1, ))
    assert_size_stride(primals_202, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (256, ), (1, ))
    assert_size_stride(primals_206, (256, ), (1, ))
    assert_size_stride(primals_207, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_213, (1024, ), (1, ))
    assert_size_stride(primals_214, (1024, ), (1, ))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, ), (1, ))
    assert_size_stride(primals_230, (1024, ), (1, ))
    assert_size_stride(primals_231, (1024, ), (1, ))
    assert_size_stride(primals_232, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (1024, ), (1, ))
    assert_size_stride(primals_246, (1024, ), (1, ))
    assert_size_stride(primals_247, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (256, ), (1, ))
    assert_size_stride(primals_267, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (256, ), (1, ))
    assert_size_stride(primals_270, (256, ), (1, ))
    assert_size_stride(primals_271, (256, ), (1, ))
    assert_size_stride(primals_272, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_273, (1024, ), (1, ))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_277, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_278, (256, ), (1, ))
    assert_size_stride(primals_279, (256, ), (1, ))
    assert_size_stride(primals_280, (256, ), (1, ))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_282, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_294, (256, ), (1, ))
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (256, ), (1, ))
    assert_size_stride(primals_297, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_298, (256, ), (1, ))
    assert_size_stride(primals_299, (256, ), (1, ))
    assert_size_stride(primals_300, (256, ), (1, ))
    assert_size_stride(primals_301, (256, ), (1, ))
    assert_size_stride(primals_302, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (256, ), (1, ))
    assert_size_stride(primals_310, (256, ), (1, ))
    assert_size_stride(primals_311, (256, ), (1, ))
    assert_size_stride(primals_312, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, ), (1, ))
    assert_size_stride(primals_315, (256, ), (1, ))
    assert_size_stride(primals_316, (256, ), (1, ))
    assert_size_stride(primals_317, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_323, (256, ), (1, ))
    assert_size_stride(primals_324, (256, ), (1, ))
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (256, ), (1, ))
    assert_size_stride(primals_330, (256, ), (1, ))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_333, (1024, ), (1, ))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_335, (1024, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_338, (256, ), (1, ))
    assert_size_stride(primals_339, (256, ), (1, ))
    assert_size_stride(primals_340, (256, ), (1, ))
    assert_size_stride(primals_341, (256, ), (1, ))
    assert_size_stride(primals_342, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, ), (1, ))
    assert_size_stride(primals_345, (256, ), (1, ))
    assert_size_stride(primals_346, (256, ), (1, ))
    assert_size_stride(primals_347, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_348, (1024, ), (1, ))
    assert_size_stride(primals_349, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_353, (256, ), (1, ))
    assert_size_stride(primals_354, (256, ), (1, ))
    assert_size_stride(primals_355, (256, ), (1, ))
    assert_size_stride(primals_356, (256, ), (1, ))
    assert_size_stride(primals_357, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_358, (256, ), (1, ))
    assert_size_stride(primals_359, (256, ), (1, ))
    assert_size_stride(primals_360, (256, ), (1, ))
    assert_size_stride(primals_361, (256, ), (1, ))
    assert_size_stride(primals_362, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (1024, ), (1, ))
    assert_size_stride(primals_365, (1024, ), (1, ))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_367, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_368, (256, ), (1, ))
    assert_size_stride(primals_369, (256, ), (1, ))
    assert_size_stride(primals_370, (256, ), (1, ))
    assert_size_stride(primals_371, (256, ), (1, ))
    assert_size_stride(primals_372, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_373, (256, ), (1, ))
    assert_size_stride(primals_374, (256, ), (1, ))
    assert_size_stride(primals_375, (256, ), (1, ))
    assert_size_stride(primals_376, (256, ), (1, ))
    assert_size_stride(primals_377, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_378, (1024, ), (1, ))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (1024, ), (1, ))
    assert_size_stride(primals_381, (1024, ), (1, ))
    assert_size_stride(primals_382, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_383, (256, ), (1, ))
    assert_size_stride(primals_384, (256, ), (1, ))
    assert_size_stride(primals_385, (256, ), (1, ))
    assert_size_stride(primals_386, (256, ), (1, ))
    assert_size_stride(primals_387, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_388, (256, ), (1, ))
    assert_size_stride(primals_389, (256, ), (1, ))
    assert_size_stride(primals_390, (256, ), (1, ))
    assert_size_stride(primals_391, (256, ), (1, ))
    assert_size_stride(primals_392, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_393, (1024, ), (1, ))
    assert_size_stride(primals_394, (1024, ), (1, ))
    assert_size_stride(primals_395, (1024, ), (1, ))
    assert_size_stride(primals_396, (1024, ), (1, ))
    assert_size_stride(primals_397, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_398, (256, ), (1, ))
    assert_size_stride(primals_399, (256, ), (1, ))
    assert_size_stride(primals_400, (256, ), (1, ))
    assert_size_stride(primals_401, (256, ), (1, ))
    assert_size_stride(primals_402, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_403, (256, ), (1, ))
    assert_size_stride(primals_404, (256, ), (1, ))
    assert_size_stride(primals_405, (256, ), (1, ))
    assert_size_stride(primals_406, (256, ), (1, ))
    assert_size_stride(primals_407, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_408, (1024, ), (1, ))
    assert_size_stride(primals_409, (1024, ), (1, ))
    assert_size_stride(primals_410, (1024, ), (1, ))
    assert_size_stride(primals_411, (1024, ), (1, ))
    assert_size_stride(primals_412, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_413, (256, ), (1, ))
    assert_size_stride(primals_414, (256, ), (1, ))
    assert_size_stride(primals_415, (256, ), (1, ))
    assert_size_stride(primals_416, (256, ), (1, ))
    assert_size_stride(primals_417, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_418, (256, ), (1, ))
    assert_size_stride(primals_419, (256, ), (1, ))
    assert_size_stride(primals_420, (256, ), (1, ))
    assert_size_stride(primals_421, (256, ), (1, ))
    assert_size_stride(primals_422, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_423, (1024, ), (1, ))
    assert_size_stride(primals_424, (1024, ), (1, ))
    assert_size_stride(primals_425, (1024, ), (1, ))
    assert_size_stride(primals_426, (1024, ), (1, ))
    assert_size_stride(primals_427, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_428, (256, ), (1, ))
    assert_size_stride(primals_429, (256, ), (1, ))
    assert_size_stride(primals_430, (256, ), (1, ))
    assert_size_stride(primals_431, (256, ), (1, ))
    assert_size_stride(primals_432, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_433, (256, ), (1, ))
    assert_size_stride(primals_434, (256, ), (1, ))
    assert_size_stride(primals_435, (256, ), (1, ))
    assert_size_stride(primals_436, (256, ), (1, ))
    assert_size_stride(primals_437, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_438, (1024, ), (1, ))
    assert_size_stride(primals_439, (1024, ), (1, ))
    assert_size_stride(primals_440, (1024, ), (1, ))
    assert_size_stride(primals_441, (1024, ), (1, ))
    assert_size_stride(primals_442, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_443, (256, ), (1, ))
    assert_size_stride(primals_444, (256, ), (1, ))
    assert_size_stride(primals_445, (256, ), (1, ))
    assert_size_stride(primals_446, (256, ), (1, ))
    assert_size_stride(primals_447, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_448, (256, ), (1, ))
    assert_size_stride(primals_449, (256, ), (1, ))
    assert_size_stride(primals_450, (256, ), (1, ))
    assert_size_stride(primals_451, (256, ), (1, ))
    assert_size_stride(primals_452, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_453, (1024, ), (1, ))
    assert_size_stride(primals_454, (1024, ), (1, ))
    assert_size_stride(primals_455, (1024, ), (1, ))
    assert_size_stride(primals_456, (1024, ), (1, ))
    assert_size_stride(primals_457, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_458, (256, ), (1, ))
    assert_size_stride(primals_459, (256, ), (1, ))
    assert_size_stride(primals_460, (256, ), (1, ))
    assert_size_stride(primals_461, (256, ), (1, ))
    assert_size_stride(primals_462, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_463, (256, ), (1, ))
    assert_size_stride(primals_464, (256, ), (1, ))
    assert_size_stride(primals_465, (256, ), (1, ))
    assert_size_stride(primals_466, (256, ), (1, ))
    assert_size_stride(primals_467, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (1024, ), (1, ))
    assert_size_stride(primals_470, (1024, ), (1, ))
    assert_size_stride(primals_471, (1024, ), (1, ))
    assert_size_stride(primals_472, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_473, (256, ), (1, ))
    assert_size_stride(primals_474, (256, ), (1, ))
    assert_size_stride(primals_475, (256, ), (1, ))
    assert_size_stride(primals_476, (256, ), (1, ))
    assert_size_stride(primals_477, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_478, (256, ), (1, ))
    assert_size_stride(primals_479, (256, ), (1, ))
    assert_size_stride(primals_480, (256, ), (1, ))
    assert_size_stride(primals_481, (256, ), (1, ))
    assert_size_stride(primals_482, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_483, (1024, ), (1, ))
    assert_size_stride(primals_484, (1024, ), (1, ))
    assert_size_stride(primals_485, (1024, ), (1, ))
    assert_size_stride(primals_486, (1024, ), (1, ))
    assert_size_stride(primals_487, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_488, (256, ), (1, ))
    assert_size_stride(primals_489, (256, ), (1, ))
    assert_size_stride(primals_490, (256, ), (1, ))
    assert_size_stride(primals_491, (256, ), (1, ))
    assert_size_stride(primals_492, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_493, (256, ), (1, ))
    assert_size_stride(primals_494, (256, ), (1, ))
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_496, (256, ), (1, ))
    assert_size_stride(primals_497, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_498, (1024, ), (1, ))
    assert_size_stride(primals_499, (1024, ), (1, ))
    assert_size_stride(primals_500, (1024, ), (1, ))
    assert_size_stride(primals_501, (1024, ), (1, ))
    assert_size_stride(primals_502, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_503, (256, ), (1, ))
    assert_size_stride(primals_504, (256, ), (1, ))
    assert_size_stride(primals_505, (256, ), (1, ))
    assert_size_stride(primals_506, (256, ), (1, ))
    assert_size_stride(primals_507, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_508, (256, ), (1, ))
    assert_size_stride(primals_509, (256, ), (1, ))
    assert_size_stride(primals_510, (256, ), (1, ))
    assert_size_stride(primals_511, (256, ), (1, ))
    assert_size_stride(primals_512, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_513, (1024, ), (1, ))
    assert_size_stride(primals_514, (1024, ), (1, ))
    assert_size_stride(primals_515, (1024, ), (1, ))
    assert_size_stride(primals_516, (1024, ), (1, ))
    assert_size_stride(primals_517, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_518, (256, ), (1, ))
    assert_size_stride(primals_519, (256, ), (1, ))
    assert_size_stride(primals_520, (256, ), (1, ))
    assert_size_stride(primals_521, (256, ), (1, ))
    assert_size_stride(primals_522, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_523, (256, ), (1, ))
    assert_size_stride(primals_524, (256, ), (1, ))
    assert_size_stride(primals_525, (256, ), (1, ))
    assert_size_stride(primals_526, (256, ), (1, ))
    assert_size_stride(primals_527, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_528, (1024, ), (1, ))
    assert_size_stride(primals_529, (1024, ), (1, ))
    assert_size_stride(primals_530, (1024, ), (1, ))
    assert_size_stride(primals_531, (1024, ), (1, ))
    assert_size_stride(primals_532, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_533, (256, ), (1, ))
    assert_size_stride(primals_534, (256, ), (1, ))
    assert_size_stride(primals_535, (256, ), (1, ))
    assert_size_stride(primals_536, (256, ), (1, ))
    assert_size_stride(primals_537, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_538, (256, ), (1, ))
    assert_size_stride(primals_539, (256, ), (1, ))
    assert_size_stride(primals_540, (256, ), (1, ))
    assert_size_stride(primals_541, (256, ), (1, ))
    assert_size_stride(primals_542, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_543, (1024, ), (1, ))
    assert_size_stride(primals_544, (1024, ), (1, ))
    assert_size_stride(primals_545, (1024, ), (1, ))
    assert_size_stride(primals_546, (1024, ), (1, ))
    assert_size_stride(primals_547, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_548, (256, ), (1, ))
    assert_size_stride(primals_549, (256, ), (1, ))
    assert_size_stride(primals_550, (256, ), (1, ))
    assert_size_stride(primals_551, (256, ), (1, ))
    assert_size_stride(primals_552, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_553, (256, ), (1, ))
    assert_size_stride(primals_554, (256, ), (1, ))
    assert_size_stride(primals_555, (256, ), (1, ))
    assert_size_stride(primals_556, (256, ), (1, ))
    assert_size_stride(primals_557, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_558, (1024, ), (1, ))
    assert_size_stride(primals_559, (1024, ), (1, ))
    assert_size_stride(primals_560, (1024, ), (1, ))
    assert_size_stride(primals_561, (1024, ), (1, ))
    assert_size_stride(primals_562, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_563, (256, ), (1, ))
    assert_size_stride(primals_564, (256, ), (1, ))
    assert_size_stride(primals_565, (256, ), (1, ))
    assert_size_stride(primals_566, (256, ), (1, ))
    assert_size_stride(primals_567, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_568, (256, ), (1, ))
    assert_size_stride(primals_569, (256, ), (1, ))
    assert_size_stride(primals_570, (256, ), (1, ))
    assert_size_stride(primals_571, (256, ), (1, ))
    assert_size_stride(primals_572, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_573, (1024, ), (1, ))
    assert_size_stride(primals_574, (1024, ), (1, ))
    assert_size_stride(primals_575, (1024, ), (1, ))
    assert_size_stride(primals_576, (1024, ), (1, ))
    assert_size_stride(primals_577, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_578, (256, ), (1, ))
    assert_size_stride(primals_579, (256, ), (1, ))
    assert_size_stride(primals_580, (256, ), (1, ))
    assert_size_stride(primals_581, (256, ), (1, ))
    assert_size_stride(primals_582, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_583, (256, ), (1, ))
    assert_size_stride(primals_584, (256, ), (1, ))
    assert_size_stride(primals_585, (256, ), (1, ))
    assert_size_stride(primals_586, (256, ), (1, ))
    assert_size_stride(primals_587, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_588, (1024, ), (1, ))
    assert_size_stride(primals_589, (1024, ), (1, ))
    assert_size_stride(primals_590, (1024, ), (1, ))
    assert_size_stride(primals_591, (1024, ), (1, ))
    assert_size_stride(primals_592, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_593, (256, ), (1, ))
    assert_size_stride(primals_594, (256, ), (1, ))
    assert_size_stride(primals_595, (256, ), (1, ))
    assert_size_stride(primals_596, (256, ), (1, ))
    assert_size_stride(primals_597, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_598, (256, ), (1, ))
    assert_size_stride(primals_599, (256, ), (1, ))
    assert_size_stride(primals_600, (256, ), (1, ))
    assert_size_stride(primals_601, (256, ), (1, ))
    assert_size_stride(primals_602, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_603, (1024, ), (1, ))
    assert_size_stride(primals_604, (1024, ), (1, ))
    assert_size_stride(primals_605, (1024, ), (1, ))
    assert_size_stride(primals_606, (1024, ), (1, ))
    assert_size_stride(primals_607, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_608, (256, ), (1, ))
    assert_size_stride(primals_609, (256, ), (1, ))
    assert_size_stride(primals_610, (256, ), (1, ))
    assert_size_stride(primals_611, (256, ), (1, ))
    assert_size_stride(primals_612, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_613, (256, ), (1, ))
    assert_size_stride(primals_614, (256, ), (1, ))
    assert_size_stride(primals_615, (256, ), (1, ))
    assert_size_stride(primals_616, (256, ), (1, ))
    assert_size_stride(primals_617, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_618, (1024, ), (1, ))
    assert_size_stride(primals_619, (1024, ), (1, ))
    assert_size_stride(primals_620, (1024, ), (1, ))
    assert_size_stride(primals_621, (1024, ), (1, ))
    assert_size_stride(primals_622, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_623, (256, ), (1, ))
    assert_size_stride(primals_624, (256, ), (1, ))
    assert_size_stride(primals_625, (256, ), (1, ))
    assert_size_stride(primals_626, (256, ), (1, ))
    assert_size_stride(primals_627, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_628, (256, ), (1, ))
    assert_size_stride(primals_629, (256, ), (1, ))
    assert_size_stride(primals_630, (256, ), (1, ))
    assert_size_stride(primals_631, (256, ), (1, ))
    assert_size_stride(primals_632, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_633, (1024, ), (1, ))
    assert_size_stride(primals_634, (1024, ), (1, ))
    assert_size_stride(primals_635, (1024, ), (1, ))
    assert_size_stride(primals_636, (1024, ), (1, ))
    assert_size_stride(primals_637, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_638, (256, ), (1, ))
    assert_size_stride(primals_639, (256, ), (1, ))
    assert_size_stride(primals_640, (256, ), (1, ))
    assert_size_stride(primals_641, (256, ), (1, ))
    assert_size_stride(primals_642, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_643, (256, ), (1, ))
    assert_size_stride(primals_644, (256, ), (1, ))
    assert_size_stride(primals_645, (256, ), (1, ))
    assert_size_stride(primals_646, (256, ), (1, ))
    assert_size_stride(primals_647, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_648, (1024, ), (1, ))
    assert_size_stride(primals_649, (1024, ), (1, ))
    assert_size_stride(primals_650, (1024, ), (1, ))
    assert_size_stride(primals_651, (1024, ), (1, ))
    assert_size_stride(primals_652, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_653, (256, ), (1, ))
    assert_size_stride(primals_654, (256, ), (1, ))
    assert_size_stride(primals_655, (256, ), (1, ))
    assert_size_stride(primals_656, (256, ), (1, ))
    assert_size_stride(primals_657, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_658, (256, ), (1, ))
    assert_size_stride(primals_659, (256, ), (1, ))
    assert_size_stride(primals_660, (256, ), (1, ))
    assert_size_stride(primals_661, (256, ), (1, ))
    assert_size_stride(primals_662, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_663, (1024, ), (1, ))
    assert_size_stride(primals_664, (1024, ), (1, ))
    assert_size_stride(primals_665, (1024, ), (1, ))
    assert_size_stride(primals_666, (1024, ), (1, ))
    assert_size_stride(primals_667, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_668, (256, ), (1, ))
    assert_size_stride(primals_669, (256, ), (1, ))
    assert_size_stride(primals_670, (256, ), (1, ))
    assert_size_stride(primals_671, (256, ), (1, ))
    assert_size_stride(primals_672, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_673, (256, ), (1, ))
    assert_size_stride(primals_674, (256, ), (1, ))
    assert_size_stride(primals_675, (256, ), (1, ))
    assert_size_stride(primals_676, (256, ), (1, ))
    assert_size_stride(primals_677, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_678, (1024, ), (1, ))
    assert_size_stride(primals_679, (1024, ), (1, ))
    assert_size_stride(primals_680, (1024, ), (1, ))
    assert_size_stride(primals_681, (1024, ), (1, ))
    assert_size_stride(primals_682, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_683, (256, ), (1, ))
    assert_size_stride(primals_684, (256, ), (1, ))
    assert_size_stride(primals_685, (256, ), (1, ))
    assert_size_stride(primals_686, (256, ), (1, ))
    assert_size_stride(primals_687, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_688, (256, ), (1, ))
    assert_size_stride(primals_689, (256, ), (1, ))
    assert_size_stride(primals_690, (256, ), (1, ))
    assert_size_stride(primals_691, (256, ), (1, ))
    assert_size_stride(primals_692, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_693, (1024, ), (1, ))
    assert_size_stride(primals_694, (1024, ), (1, ))
    assert_size_stride(primals_695, (1024, ), (1, ))
    assert_size_stride(primals_696, (1024, ), (1, ))
    assert_size_stride(primals_697, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_698, (256, ), (1, ))
    assert_size_stride(primals_699, (256, ), (1, ))
    assert_size_stride(primals_700, (256, ), (1, ))
    assert_size_stride(primals_701, (256, ), (1, ))
    assert_size_stride(primals_702, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_703, (256, ), (1, ))
    assert_size_stride(primals_704, (256, ), (1, ))
    assert_size_stride(primals_705, (256, ), (1, ))
    assert_size_stride(primals_706, (256, ), (1, ))
    assert_size_stride(primals_707, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_708, (1024, ), (1, ))
    assert_size_stride(primals_709, (1024, ), (1, ))
    assert_size_stride(primals_710, (1024, ), (1, ))
    assert_size_stride(primals_711, (1024, ), (1, ))
    assert_size_stride(primals_712, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_713, (256, ), (1, ))
    assert_size_stride(primals_714, (256, ), (1, ))
    assert_size_stride(primals_715, (256, ), (1, ))
    assert_size_stride(primals_716, (256, ), (1, ))
    assert_size_stride(primals_717, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_718, (256, ), (1, ))
    assert_size_stride(primals_719, (256, ), (1, ))
    assert_size_stride(primals_720, (256, ), (1, ))
    assert_size_stride(primals_721, (256, ), (1, ))
    assert_size_stride(primals_722, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_723, (1024, ), (1, ))
    assert_size_stride(primals_724, (1024, ), (1, ))
    assert_size_stride(primals_725, (1024, ), (1, ))
    assert_size_stride(primals_726, (1024, ), (1, ))
    assert_size_stride(primals_727, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_728, (512, ), (1, ))
    assert_size_stride(primals_729, (512, ), (1, ))
    assert_size_stride(primals_730, (512, ), (1, ))
    assert_size_stride(primals_731, (512, ), (1, ))
    assert_size_stride(primals_732, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_733, (512, ), (1, ))
    assert_size_stride(primals_734, (512, ), (1, ))
    assert_size_stride(primals_735, (512, ), (1, ))
    assert_size_stride(primals_736, (512, ), (1, ))
    assert_size_stride(primals_737, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_738, (2048, ), (1, ))
    assert_size_stride(primals_739, (2048, ), (1, ))
    assert_size_stride(primals_740, (2048, ), (1, ))
    assert_size_stride(primals_741, (2048, ), (1, ))
    assert_size_stride(primals_742, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_743, (2048, ), (1, ))
    assert_size_stride(primals_744, (2048, ), (1, ))
    assert_size_stride(primals_745, (2048, ), (1, ))
    assert_size_stride(primals_746, (2048, ), (1, ))
    assert_size_stride(primals_747, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_748, (512, ), (1, ))
    assert_size_stride(primals_749, (512, ), (1, ))
    assert_size_stride(primals_750, (512, ), (1, ))
    assert_size_stride(primals_751, (512, ), (1, ))
    assert_size_stride(primals_752, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_753, (512, ), (1, ))
    assert_size_stride(primals_754, (512, ), (1, ))
    assert_size_stride(primals_755, (512, ), (1, ))
    assert_size_stride(primals_756, (512, ), (1, ))
    assert_size_stride(primals_757, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_758, (2048, ), (1, ))
    assert_size_stride(primals_759, (2048, ), (1, ))
    assert_size_stride(primals_760, (2048, ), (1, ))
    assert_size_stride(primals_761, (2048, ), (1, ))
    assert_size_stride(primals_762, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_763, (512, ), (1, ))
    assert_size_stride(primals_764, (512, ), (1, ))
    assert_size_stride(primals_765, (512, ), (1, ))
    assert_size_stride(primals_766, (512, ), (1, ))
    assert_size_stride(primals_767, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_768, (512, ), (1, ))
    assert_size_stride(primals_769, (512, ), (1, ))
    assert_size_stride(primals_770, (512, ), (1, ))
    assert_size_stride(primals_771, (512, ), (1, ))
    assert_size_stride(primals_772, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_773, (2048, ), (1, ))
    assert_size_stride(primals_774, (2048, ), (1, ))
    assert_size_stride(primals_775, (2048, ), (1, ))
    assert_size_stride(primals_776, (2048, ), (1, ))
    assert_size_stride(primals_777, (256, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_778, (256, ), (1, ))
    assert_size_stride(primals_779, (256, ), (1, ))
    assert_size_stride(primals_780, (256, ), (1, ))
    assert_size_stride(primals_781, (256, ), (1, ))
    assert_size_stride(primals_782, (256, ), (1, ))
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
        buf2 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf1, buf2, buf3, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf5 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_8, primals_9, primals_10, primals_11, buf5, 65536, grid=grid(65536), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf7 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, primals_13, primals_14, primals_15, primals_16, buf7, 65536, grid=grid(65536), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf2, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf10 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_7, input_6, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf11, buf8, primals_18, primals_19, primals_20, primals_21, buf9, primals_23, primals_24, primals_25, primals_26, 262144, grid=grid(262144), stream=stream0)
        del primals_21
        del primals_26
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf13 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf12, primals_28, primals_29, primals_30, primals_31, buf13, 65536, grid=grid(65536), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf15 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf14, primals_33, primals_34, primals_35, primals_36, buf15, 65536, grid=grid(65536), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf16, primals_38, primals_39, primals_40, primals_41, buf11, buf17, 262144, grid=grid(262144), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf18, primals_43, primals_44, primals_45, primals_46, buf19, 65536, grid=grid(65536), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf21 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf20, primals_48, primals_49, primals_50, primals_51, buf21, 65536, grid=grid(65536), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf23 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf22, primals_53, primals_54, primals_55, primals_56, buf17, buf23, 262144, grid=grid(262144), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf25 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf24, primals_58, primals_59, primals_60, primals_61, buf25, 131072, grid=grid(131072), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_62, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf27 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf26, primals_63, primals_64, primals_65, primals_66, buf27, 32768, grid=grid(32768), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 512, 8, 8), (32768, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf23, primals_72, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf30 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [out_37, input_8, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf31, buf28, primals_68, primals_69, primals_70, primals_71, buf29, primals_73, primals_74, primals_75, primals_76, 131072, grid=grid(131072), stream=stream0)
        del primals_71
        del primals_76
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf33 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf32, primals_78, primals_79, primals_80, primals_81, buf33, 32768, grid=grid(32768), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf35 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf34, primals_83, primals_84, primals_85, primals_86, buf35, 32768, grid=grid(32768), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf37 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_47, out_48, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf36, primals_88, primals_89, primals_90, primals_91, buf31, buf37, 131072, grid=grid(131072), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf39 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf38, primals_93, primals_94, primals_95, primals_96, buf39, 32768, grid=grid(32768), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf41 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf40, primals_98, primals_99, primals_100, primals_101, buf41, 32768, grid=grid(32768), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf43 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf42, primals_103, primals_104, primals_105, primals_106, buf37, buf43, 131072, grid=grid(131072), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf45 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf44, primals_108, primals_109, primals_110, primals_111, buf45, 32768, grid=grid(32768), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf47 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf46, primals_113, primals_114, primals_115, primals_116, buf47, 32768, grid=grid(32768), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf49 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_67, out_68, out_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf48, primals_118, primals_119, primals_120, primals_121, buf43, buf49, 131072, grid=grid(131072), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf51 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf50, primals_123, primals_124, primals_125, primals_126, buf51, 32768, grid=grid(32768), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf53 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf52, primals_128, primals_129, primals_130, primals_131, buf53, 32768, grid=grid(32768), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf55 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_77, out_78, out_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf54, primals_133, primals_134, primals_135, primals_136, buf49, buf55, 131072, grid=grid(131072), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf57 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf56, primals_138, primals_139, primals_140, primals_141, buf57, 32768, grid=grid(32768), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf59 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf58, primals_143, primals_144, primals_145, primals_146, buf59, 32768, grid=grid(32768), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf61 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_87, out_88, out_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf60, primals_148, primals_149, primals_150, primals_151, buf55, buf61, 131072, grid=grid(131072), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf63 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf62, primals_153, primals_154, primals_155, primals_156, buf63, 32768, grid=grid(32768), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf65 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf64, primals_158, primals_159, primals_160, primals_161, buf65, 32768, grid=grid(32768), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf67 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf66, primals_163, primals_164, primals_165, primals_166, buf61, buf67, 131072, grid=grid(131072), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf69 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf68, primals_168, primals_169, primals_170, primals_171, buf69, 32768, grid=grid(32768), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf71 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf70, primals_173, primals_174, primals_175, primals_176, buf71, 32768, grid=grid(32768), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf73 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf72, primals_178, primals_179, primals_180, primals_181, buf67, buf73, 131072, grid=grid(131072), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf75 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf74, primals_183, primals_184, primals_185, primals_186, buf75, 65536, grid=grid(65536), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf77 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf76, primals_188, primals_189, primals_190, primals_191, buf77, 65536, grid=grid(65536), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 1024, 8, 8), (65536, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf73, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf80 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [out_117, input_10, out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf81, buf78, primals_193, primals_194, primals_195, primals_196, buf79, primals_198, primals_199, primals_200, primals_201, 262144, grid=grid(262144), stream=stream0)
        del primals_196
        del primals_201
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf83 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf82, primals_203, primals_204, primals_205, primals_206, buf83, 65536, grid=grid(65536), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_207, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf85 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf84, primals_208, primals_209, primals_210, primals_211, buf85, 65536, grid=grid(65536), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf87 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_127, out_128, out_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf86, primals_213, primals_214, primals_215, primals_216, buf81, buf87, 262144, grid=grid(262144), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf89 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf88, primals_218, primals_219, primals_220, primals_221, buf89, 65536, grid=grid(65536), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_222, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf91 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf90, primals_223, primals_224, primals_225, primals_226, buf91, 65536, grid=grid(65536), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf93 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_137, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf92, primals_228, primals_229, primals_230, primals_231, buf87, buf93, 262144, grid=grid(262144), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf95 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf94, primals_233, primals_234, primals_235, primals_236, buf95, 65536, grid=grid(65536), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_237, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf97 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf96, primals_238, primals_239, primals_240, primals_241, buf97, 65536, grid=grid(65536), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf99 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf98, primals_243, primals_244, primals_245, primals_246, buf93, buf99, 262144, grid=grid(262144), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf101 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf100, primals_248, primals_249, primals_250, primals_251, buf101, 65536, grid=grid(65536), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf103 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf102, primals_253, primals_254, primals_255, primals_256, buf103, 65536, grid=grid(65536), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf105 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_157, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf104, primals_258, primals_259, primals_260, primals_261, buf99, buf105, 262144, grid=grid(262144), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf107 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf106, primals_263, primals_264, primals_265, primals_266, buf107, 65536, grid=grid(65536), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_267, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf109 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_164, out_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf108, primals_268, primals_269, primals_270, primals_271, buf109, 65536, grid=grid(65536), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf111 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_167, out_168, out_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf110, primals_273, primals_274, primals_275, primals_276, buf105, buf111, 262144, grid=grid(262144), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf113 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_171, out_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf112, primals_278, primals_279, primals_280, primals_281, buf113, 65536, grid=grid(65536), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [out_173], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_282, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf115 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf114, primals_283, primals_284, primals_285, primals_286, buf115, 65536, grid=grid(65536), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf117 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_177, out_178, out_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf116, primals_288, primals_289, primals_290, primals_291, buf111, buf117, 262144, grid=grid(262144), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf119 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_181, out_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf118, primals_293, primals_294, primals_295, primals_296, buf119, 65536, grid=grid(65536), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [out_183], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_297, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf121 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_184, out_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf120, primals_298, primals_299, primals_300, primals_301, buf121, 65536, grid=grid(65536), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [out_186], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf123 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_187, out_188, out_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf122, primals_303, primals_304, primals_305, primals_306, buf117, buf123, 262144, grid=grid(262144), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf125 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_191, out_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf124, primals_308, primals_309, primals_310, primals_311, buf125, 65536, grid=grid(65536), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [out_193], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf127 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_194, out_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf126, primals_313, primals_314, primals_315, primals_316, buf127, 65536, grid=grid(65536), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf129 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf128, primals_318, primals_319, primals_320, primals_321, buf123, buf129, 262144, grid=grid(262144), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf131 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf130, primals_323, primals_324, primals_325, primals_326, buf131, 65536, grid=grid(65536), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [out_203], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_327, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf133 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_204, out_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf132, primals_328, primals_329, primals_330, primals_331, buf133, 65536, grid=grid(65536), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [out_206], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf135 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_207, out_208, out_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf134, primals_333, primals_334, primals_335, primals_336, buf129, buf135, 262144, grid=grid(262144), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [out_210], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf137 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_211, out_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf136, primals_338, primals_339, primals_340, primals_341, buf137, 65536, grid=grid(65536), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [out_213], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_342, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf139 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_214, out_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf138, primals_343, primals_344, primals_345, primals_346, buf139, 65536, grid=grid(65536), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf141 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_217, out_218, out_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf140, primals_348, primals_349, primals_350, primals_351, buf135, buf141, 262144, grid=grid(262144), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf143 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_221, out_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf142, primals_353, primals_354, primals_355, primals_356, buf143, 65536, grid=grid(65536), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [out_223], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_357, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf145 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_224, out_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf144, primals_358, primals_359, primals_360, primals_361, buf145, 65536, grid=grid(65536), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [out_226], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf147 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_227, out_228, out_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf146, primals_363, primals_364, primals_365, primals_366, buf141, buf147, 262144, grid=grid(262144), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [out_230], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf149 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_231, out_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf148, primals_368, primals_369, primals_370, primals_371, buf149, 65536, grid=grid(65536), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [out_233], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf151 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_234, out_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf150, primals_373, primals_374, primals_375, primals_376, buf151, 65536, grid=grid(65536), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf153 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_237, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf152, primals_378, primals_379, primals_380, primals_381, buf147, buf153, 262144, grid=grid(262144), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [out_240], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf155 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf154, primals_383, primals_384, primals_385, primals_386, buf155, 65536, grid=grid(65536), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_387, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf157 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_244, out_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf156, primals_388, primals_389, primals_390, primals_391, buf157, 65536, grid=grid(65536), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [out_246], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf159 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_247, out_248, out_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf158, primals_393, primals_394, primals_395, primals_396, buf153, buf159, 262144, grid=grid(262144), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf161 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_251, out_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf160, primals_398, primals_399, primals_400, primals_401, buf161, 65536, grid=grid(65536), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [out_253], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_402, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf163 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_254, out_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf162, primals_403, primals_404, primals_405, primals_406, buf163, 65536, grid=grid(65536), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [out_256], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf165 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_257, out_258, out_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf164, primals_408, primals_409, primals_410, primals_411, buf159, buf165, 262144, grid=grid(262144), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_412, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf167 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_261, out_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf166, primals_413, primals_414, primals_415, primals_416, buf167, 65536, grid=grid(65536), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [out_263], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_417, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf169 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_264, out_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf168, primals_418, primals_419, primals_420, primals_421, buf169, 65536, grid=grid(65536), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [out_266], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf171 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_267, out_268, out_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf170, primals_423, primals_424, primals_425, primals_426, buf165, buf171, 262144, grid=grid(262144), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [out_270], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf173 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_271, out_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf172, primals_428, primals_429, primals_430, primals_431, buf173, 65536, grid=grid(65536), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [out_273], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf175 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_274, out_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf174, primals_433, primals_434, primals_435, primals_436, buf175, 65536, grid=grid(65536), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [out_276], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf177 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_277, out_278, out_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf176, primals_438, primals_439, primals_440, primals_441, buf171, buf177, 262144, grid=grid(262144), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [out_280], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf179 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_281, out_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf178, primals_443, primals_444, primals_445, primals_446, buf179, 65536, grid=grid(65536), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [out_283], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_447, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf181 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_284, out_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf180, primals_448, primals_449, primals_450, primals_451, buf181, 65536, grid=grid(65536), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [out_286], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf183 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_287, out_288, out_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf182, primals_453, primals_454, primals_455, primals_456, buf177, buf183, 262144, grid=grid(262144), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [out_290], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf185 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_291, out_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf184, primals_458, primals_459, primals_460, primals_461, buf185, 65536, grid=grid(65536), stream=stream0)
        del primals_461
        # Topologically Sorted Source Nodes: [out_293], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_462, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf187 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_294, out_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf186, primals_463, primals_464, primals_465, primals_466, buf187, 65536, grid=grid(65536), stream=stream0)
        del primals_466
        # Topologically Sorted Source Nodes: [out_296], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf189 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_297, out_298, out_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf188, primals_468, primals_469, primals_470, primals_471, buf183, buf189, 262144, grid=grid(262144), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf191 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_301, out_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf190, primals_473, primals_474, primals_475, primals_476, buf191, 65536, grid=grid(65536), stream=stream0)
        del primals_476
        # Topologically Sorted Source Nodes: [out_303], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_477, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf193 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_304, out_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf192, primals_478, primals_479, primals_480, primals_481, buf193, 65536, grid=grid(65536), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [out_306], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf195 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_307, out_308, out_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf194, primals_483, primals_484, primals_485, primals_486, buf189, buf195, 262144, grid=grid(262144), stream=stream0)
        del primals_486
        # Topologically Sorted Source Nodes: [out_310], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf197 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_311, out_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf196, primals_488, primals_489, primals_490, primals_491, buf197, 65536, grid=grid(65536), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [out_313], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf199 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_314, out_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf198, primals_493, primals_494, primals_495, primals_496, buf199, 65536, grid=grid(65536), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [out_316], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_497, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf201 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_317, out_318, out_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf200, primals_498, primals_499, primals_500, primals_501, buf195, buf201, 262144, grid=grid(262144), stream=stream0)
        del primals_501
        # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf203 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf202, primals_503, primals_504, primals_505, primals_506, buf203, 65536, grid=grid(65536), stream=stream0)
        del primals_506
        # Topologically Sorted Source Nodes: [out_323], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_507, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf205 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_324, out_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf204, primals_508, primals_509, primals_510, primals_511, buf205, 65536, grid=grid(65536), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [out_326], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf207 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_327, out_328, out_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf206, primals_513, primals_514, primals_515, primals_516, buf201, buf207, 262144, grid=grid(262144), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [out_330], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf209 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_331, out_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf208, primals_518, primals_519, primals_520, primals_521, buf209, 65536, grid=grid(65536), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [out_333], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_522, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf211 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_334, out_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf210, primals_523, primals_524, primals_525, primals_526, buf211, 65536, grid=grid(65536), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [out_336], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_527, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf213 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_337, out_338, out_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf212, primals_528, primals_529, primals_530, primals_531, buf207, buf213, 262144, grid=grid(262144), stream=stream0)
        del primals_531
        # Topologically Sorted Source Nodes: [out_340], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf215 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_341, out_342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf214, primals_533, primals_534, primals_535, primals_536, buf215, 65536, grid=grid(65536), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [out_343], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_537, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf217 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_344, out_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf216, primals_538, primals_539, primals_540, primals_541, buf217, 65536, grid=grid(65536), stream=stream0)
        del primals_541
        # Topologically Sorted Source Nodes: [out_346], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf219 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_347, out_348, out_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf218, primals_543, primals_544, primals_545, primals_546, buf213, buf219, 262144, grid=grid(262144), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [out_350], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_547, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf221 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_351, out_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf220, primals_548, primals_549, primals_550, primals_551, buf221, 65536, grid=grid(65536), stream=stream0)
        del primals_551
        # Topologically Sorted Source Nodes: [out_353], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf223 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_354, out_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf222, primals_553, primals_554, primals_555, primals_556, buf223, 65536, grid=grid(65536), stream=stream0)
        del primals_556
        # Topologically Sorted Source Nodes: [out_356], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_557, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf225 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_357, out_358, out_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf224, primals_558, primals_559, primals_560, primals_561, buf219, buf225, 262144, grid=grid(262144), stream=stream0)
        del primals_561
        # Topologically Sorted Source Nodes: [out_360], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_562, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf227 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_361, out_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf226, primals_563, primals_564, primals_565, primals_566, buf227, 65536, grid=grid(65536), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [out_363], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_567, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf229 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_364, out_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf228, primals_568, primals_569, primals_570, primals_571, buf229, 65536, grid=grid(65536), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [out_366], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_572, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf231 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_367, out_368, out_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf230, primals_573, primals_574, primals_575, primals_576, buf225, buf231, 262144, grid=grid(262144), stream=stream0)
        del primals_576
        # Topologically Sorted Source Nodes: [out_370], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_577, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf233 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_371, out_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf232, primals_578, primals_579, primals_580, primals_581, buf233, 65536, grid=grid(65536), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [out_373], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_582, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf235 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_374, out_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf234, primals_583, primals_584, primals_585, primals_586, buf235, 65536, grid=grid(65536), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [out_376], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_587, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf237 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_377, out_378, out_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf236, primals_588, primals_589, primals_590, primals_591, buf231, buf237, 262144, grid=grid(262144), stream=stream0)
        del primals_591
        # Topologically Sorted Source Nodes: [out_380], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_592, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf239 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf238, primals_593, primals_594, primals_595, primals_596, buf239, 65536, grid=grid(65536), stream=stream0)
        del primals_596
        # Topologically Sorted Source Nodes: [out_383], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_597, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf241 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_384, out_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf240, primals_598, primals_599, primals_600, primals_601, buf241, 65536, grid=grid(65536), stream=stream0)
        del primals_601
        # Topologically Sorted Source Nodes: [out_386], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_602, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf243 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_387, out_388, out_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf242, primals_603, primals_604, primals_605, primals_606, buf237, buf243, 262144, grid=grid(262144), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [out_390], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_607, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf245 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_391, out_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf244, primals_608, primals_609, primals_610, primals_611, buf245, 65536, grid=grid(65536), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [out_393], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_612, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf247 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_394, out_395], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf246, primals_613, primals_614, primals_615, primals_616, buf247, 65536, grid=grid(65536), stream=stream0)
        del primals_616
        # Topologically Sorted Source Nodes: [out_396], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_617, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf249 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_397, out_398, out_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf248, primals_618, primals_619, primals_620, primals_621, buf243, buf249, 262144, grid=grid(262144), stream=stream0)
        del primals_621
        # Topologically Sorted Source Nodes: [out_400], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_622, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf251 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_401, out_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf250, primals_623, primals_624, primals_625, primals_626, buf251, 65536, grid=grid(65536), stream=stream0)
        del primals_626
        # Topologically Sorted Source Nodes: [out_403], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_627, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf253 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_404, out_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf252, primals_628, primals_629, primals_630, primals_631, buf253, 65536, grid=grid(65536), stream=stream0)
        del primals_631
        # Topologically Sorted Source Nodes: [out_406], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_632, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf255 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_407, out_408, out_409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf254, primals_633, primals_634, primals_635, primals_636, buf249, buf255, 262144, grid=grid(262144), stream=stream0)
        del primals_636
        # Topologically Sorted Source Nodes: [out_410], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_637, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf257 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_411, out_412], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf256, primals_638, primals_639, primals_640, primals_641, buf257, 65536, grid=grid(65536), stream=stream0)
        del primals_641
        # Topologically Sorted Source Nodes: [out_413], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_642, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf259 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_414, out_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf258, primals_643, primals_644, primals_645, primals_646, buf259, 65536, grid=grid(65536), stream=stream0)
        del primals_646
        # Topologically Sorted Source Nodes: [out_416], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_647, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf261 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_417, out_418, out_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf260, primals_648, primals_649, primals_650, primals_651, buf255, buf261, 262144, grid=grid(262144), stream=stream0)
        del primals_651
        # Topologically Sorted Source Nodes: [out_420], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_652, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf263 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_421, out_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf262, primals_653, primals_654, primals_655, primals_656, buf263, 65536, grid=grid(65536), stream=stream0)
        del primals_656
        # Topologically Sorted Source Nodes: [out_423], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_657, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf265 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_424, out_425], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf264, primals_658, primals_659, primals_660, primals_661, buf265, 65536, grid=grid(65536), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [out_426], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_662, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf267 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_427, out_428, out_429], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf266, primals_663, primals_664, primals_665, primals_666, buf261, buf267, 262144, grid=grid(262144), stream=stream0)
        del primals_666
        # Topologically Sorted Source Nodes: [out_430], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_667, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf269 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_431, out_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf268, primals_668, primals_669, primals_670, primals_671, buf269, 65536, grid=grid(65536), stream=stream0)
        del primals_671
        # Topologically Sorted Source Nodes: [out_433], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_672, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf271 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_434, out_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf270, primals_673, primals_674, primals_675, primals_676, buf271, 65536, grid=grid(65536), stream=stream0)
        del primals_676
        # Topologically Sorted Source Nodes: [out_436], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_677, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf273 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_437, out_438, out_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf272, primals_678, primals_679, primals_680, primals_681, buf267, buf273, 262144, grid=grid(262144), stream=stream0)
        del primals_681
        # Topologically Sorted Source Nodes: [out_440], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_682, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf275 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_441, out_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf274, primals_683, primals_684, primals_685, primals_686, buf275, 65536, grid=grid(65536), stream=stream0)
        del primals_686
        # Topologically Sorted Source Nodes: [out_443], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_687, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf277 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_444, out_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf276, primals_688, primals_689, primals_690, primals_691, buf277, 65536, grid=grid(65536), stream=stream0)
        del primals_691
        # Topologically Sorted Source Nodes: [out_446], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_692, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf279 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_447, out_448, out_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf278, primals_693, primals_694, primals_695, primals_696, buf273, buf279, 262144, grid=grid(262144), stream=stream0)
        del primals_696
        # Topologically Sorted Source Nodes: [out_450], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_697, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf281 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_451, out_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf280, primals_698, primals_699, primals_700, primals_701, buf281, 65536, grid=grid(65536), stream=stream0)
        del primals_701
        # Topologically Sorted Source Nodes: [out_453], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_702, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf283 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_454, out_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf282, primals_703, primals_704, primals_705, primals_706, buf283, 65536, grid=grid(65536), stream=stream0)
        del primals_706
        # Topologically Sorted Source Nodes: [out_456], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_707, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf285 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_457, out_458, out_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf284, primals_708, primals_709, primals_710, primals_711, buf279, buf285, 262144, grid=grid(262144), stream=stream0)
        del primals_711
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_712, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf287 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_461, out_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf286, primals_713, primals_714, primals_715, primals_716, buf287, 65536, grid=grid(65536), stream=stream0)
        del primals_716
        # Topologically Sorted Source Nodes: [out_463], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_717, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf289 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_464, out_465], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf288, primals_718, primals_719, primals_720, primals_721, buf289, 65536, grid=grid(65536), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [out_466], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_722, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 1024, 8, 8), (65536, 64, 8, 1))
        buf291 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_467, out_468, out_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf290, primals_723, primals_724, primals_725, primals_726, buf285, buf291, 262144, grid=grid(262144), stream=stream0)
        del primals_726
        # Topologically Sorted Source Nodes: [out_470], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_727, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf293 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_471, out_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf292, primals_728, primals_729, primals_730, primals_731, buf293, 131072, grid=grid(131072), stream=stream0)
        del primals_731
        # Topologically Sorted Source Nodes: [out_473], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_732, stride=(1, 1), padding=(5, 5), dilation=(5, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf295 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_474, out_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf294, primals_733, primals_734, primals_735, primals_736, buf295, 131072, grid=grid(131072), stream=stream0)
        del primals_736
        # Topologically Sorted Source Nodes: [out_476], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_737, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 2048, 8, 8), (131072, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf291, primals_742, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 2048, 8, 8), (131072, 64, 8, 1))
        buf298 = empty_strided_cuda((4, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [out_477, input_12, out_478, out_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf299, buf296, primals_738, primals_739, primals_740, primals_741, buf297, primals_743, primals_744, primals_745, primals_746, 524288, grid=grid(524288), stream=stream0)
        del primals_741
        del primals_746
        # Topologically Sorted Source Nodes: [out_480], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_747, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf301 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_481, out_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf300, primals_748, primals_749, primals_750, primals_751, buf301, 131072, grid=grid(131072), stream=stream0)
        del primals_751
        # Topologically Sorted Source Nodes: [out_483], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_752, stride=(1, 1), padding=(9, 9), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf303 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_484, out_485], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf302, primals_753, primals_754, primals_755, primals_756, buf303, 131072, grid=grid(131072), stream=stream0)
        del primals_756
        # Topologically Sorted Source Nodes: [out_486], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_757, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 2048, 8, 8), (131072, 64, 8, 1))
        buf305 = empty_strided_cuda((4, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_487, out_488, out_489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf304, primals_758, primals_759, primals_760, primals_761, buf299, buf305, 524288, grid=grid(524288), stream=stream0)
        del primals_761
        # Topologically Sorted Source Nodes: [out_490], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_762, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf307 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_491, out_492], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf306, primals_763, primals_764, primals_765, primals_766, buf307, 131072, grid=grid(131072), stream=stream0)
        del primals_766
        # Topologically Sorted Source Nodes: [out_493], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_767, stride=(1, 1), padding=(17, 17), dilation=(17, 17), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf309 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_494, out_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf308, primals_768, primals_769, primals_770, primals_771, buf309, 131072, grid=grid(131072), stream=stream0)
        del primals_771
        # Topologically Sorted Source Nodes: [out_496], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 2048, 8, 8), (131072, 64, 8, 1))
        buf311 = empty_strided_cuda((4, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_497, out_498, out_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf310, primals_773, primals_774, primals_775, primals_776, buf305, buf311, 524288, grid=grid(524288), stream=stream0)
        del primals_776
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_777, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf313 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf313, primals_778, 65536, grid=grid(65536), stream=stream0)
        del primals_778
        buf314 = empty_strided_cuda((4, 4, 8, 8, 8, 8), (16384, 4096, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pixel_shuffle]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pixel_shuffle_16.run(buf313, primals_779, primals_780, primals_781, primals_782, buf314, 8192, 8, grid=grid(8192, 8), stream=stream0)
    return (reinterpret_tensor(buf314, (4, 4, 64, 64), (16384, 4096, 64, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_465, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_525, primals_527, primals_528, primals_529, primals_530, primals_532, primals_533, primals_534, primals_535, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_562, primals_563, primals_564, primals_565, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_595, primals_597, primals_598, primals_599, primals_600, primals_602, primals_603, primals_604, primals_605, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_627, primals_628, primals_629, primals_630, primals_632, primals_633, primals_634, primals_635, primals_637, primals_638, primals_639, primals_640, primals_642, primals_643, primals_644, primals_645, primals_647, primals_648, primals_649, primals_650, primals_652, primals_653, primals_654, primals_655, primals_657, primals_658, primals_659, primals_660, primals_662, primals_663, primals_664, primals_665, primals_667, primals_668, primals_669, primals_670, primals_672, primals_673, primals_674, primals_675, primals_677, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_685, primals_687, primals_688, primals_689, primals_690, primals_692, primals_693, primals_694, primals_695, primals_697, primals_698, primals_699, primals_700, primals_702, primals_703, primals_704, primals_705, primals_707, primals_708, primals_709, primals_710, primals_712, primals_713, primals_714, primals_715, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_735, primals_737, primals_738, primals_739, primals_740, primals_742, primals_743, primals_744, primals_745, primals_747, primals_748, primals_749, primals_750, primals_752, primals_753, primals_754, primals_755, primals_757, primals_758, primals_759, primals_760, primals_762, primals_763, primals_764, primals_765, primals_767, primals_768, primals_769, primals_770, primals_772, primals_773, primals_774, primals_775, primals_777, primals_779, primals_780, primals_781, primals_782, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf246, buf247, buf248, buf249, buf250, buf251, buf252, buf253, buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf273, buf274, buf275, buf276, buf277, buf278, buf279, buf280, buf281, buf282, buf283, buf284, buf285, buf286, buf287, buf288, buf289, buf290, buf291, buf292, buf293, buf294, buf295, buf296, buf297, buf299, buf300, buf301, buf302, buf303, buf304, buf305, buf306, buf307, buf308, buf309, buf310, buf311, buf313, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((256, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
