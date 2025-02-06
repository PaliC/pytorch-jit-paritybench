# AOT ID: ['0_forward']
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
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
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
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_3 => getitem, getitem_1
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
# Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_9, mul_13, mul_14, sub_4
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
# Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_32, mul_43, mul_44, sub_14
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
# Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_71 => add_56, mul_73, mul_74, sub_24
#   out_72 => relu_22
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
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


# kernel path: inductor_cache/x2/cx2inxu7r47zonxvqgltrri7zxxp2l5wwvrqc4wenh5rt6xtxtw5.py
# Topologically Sorted Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_74 => add_58, mul_76, mul_77, sub_25
#   out_75 => relu_23
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
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
    x3 = xindex
    x1 = ((xindex // 16) % 256)
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


# kernel path: inductor_cache/zw/czweqkkvn3m53ycxvqfe7npsjog5pfekgjmubfmii3aeza3suaoz.py
# Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_62, mul_82, mul_83, sub_27
#   out_77 => add_60, mul_79, mul_80, sub_26
#   out_78 => add_63
#   out_79 => relu_24
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %add_62), kwargs = {})
#   %relu_24 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yq/cyqgzgpjqqhrdxykyztxa6pi72hsduwfu6bxs3i5s46ele4aey5j.py
# Topologically Sorted Source Nodes: [out_87, out_88, out_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_87 => add_69, mul_91, mul_92, sub_30
#   out_88 => add_70
#   out_89 => relu_27
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %relu_24), kwargs = {})
#   %relu_27 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ou/couka5jck3crti3va3x5x4wanr2atiyxdcoy2zoiqce3qfpo2swu.py
# Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_131 => add_100, mul_130, mul_131, sub_43
#   out_132 => relu_40
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_100,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/uf/cufskar5wj3o53y3p6srp6b7ac3j26mumge566tzrazsigxndkcf.py
# Topologically Sorted Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_134 => add_102, mul_133, mul_134, sub_44
#   out_135 => relu_41
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_357), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_359), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_102,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
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


# kernel path: inductor_cache/6z/c6zua2fmphqqg527kvlyc2wjqrvpc4affwsqseqt6oqvhsejao3d.py
# Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_106, mul_139, mul_140, sub_46
#   out_137 => add_104, mul_136, mul_137, sub_45
#   out_138 => add_107
#   out_139 => relu_42
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_373), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_375), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %add_106), kwargs = {})
#   %relu_42 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 2048)
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


# kernel path: inductor_cache/jb/cjbmtr5bi4whmr26fqp36jxccwtle62yskr7yzd44bbptgqqx2xd.py
# Topologically Sorted Source Nodes: [out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_147 => add_113, mul_148, mul_149, sub_49
#   out_148 => add_114
#   out_149 => relu_45
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %relu_42), kwargs = {})
#   %relu_45 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_114,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 2048)
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


# kernel path: inductor_cache/ph/cph4kn5lt4mzcekndo3iyyjzfhw3yjdknnbkssllkoepbw6uvcs7.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_5 => add_123, mul_160, mul_161, sub_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_123 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 1024)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/4y/c4yswgh5dhbqrvvqeqdvgse4gjsicpqbfre3dhonrmgxckxywila.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_123, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_18 = async_compile.triton('triton_poi_fused_constant_pad_nd_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-3) + x0 + 2*x1 + 4*x2), tmp10, other=0.0)
    tl.store(out_ptr0 + (x4), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/wy/cwyldlyayw3rveufyitsaer35ayfq2lpo2lo4utydcso2t3d4ous.py
# Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_1 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_123, [1, 1, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_19 = async_compile.triton('triton_poi_fused_constant_pad_nd_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 3)
    x0 = (xindex % 4)
    x2 = xindex // 12
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-3) + x0 + 2*x1 + 4*x2), tmp8, other=0.0)
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/47/c47rtubxezjcgg3dy3fssd3jd7kk5mr7bnjbog4gtz7ezgwhkuhc.py
# Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_2 => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_123, [1, 0, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_20 = async_compile.triton('triton_poi_fused_constant_pad_nd_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 3) % 4)
    x0 = (xindex % 3)
    x2 = xindex // 12
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-3) + x0 + 2*x1 + 4*x2), tmp8, other=0.0)
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/oa/coabnjsesv5d23dfvxlmgd4ajn6wclzwh2g3tkt5vtjkbchrbf2n.py
# Topologically Sorted Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_3 => constant_pad_nd_3
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_123, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_21 = async_compile.triton('triton_poi_fused_constant_pad_nd_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-3) + x0 + 2*x1 + 4*x2), tmp5, other=0.0)
    tl.store(out_ptr0 + (x4), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/3b/c3bpzitjverlsivrd2gtqaqwtjnafomxv4eppgmtiznre44334ju.py
# Topologically Sorted Source Nodes: [stack_2], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view_3], 2), kwargs = {})
triton_poi_fused_stack_22 = async_compile.triton('triton_poi_fused_stack_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex // 16
    x2 = ((xindex // 16) % 512)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 2*((x0 % 2)) + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (2*(2*((x0 % 2)) + (x1)) + 4*x4 + (x0 // 2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 4, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (2*((-2) + 2*((x0 % 2)) + (x1)) + 4*x4 + (x0 // 2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp15, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 4, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = 2*((x0 % 2)) + ((-2) + x1)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 2, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp28
    tmp37 = tl.load(in_ptr4 + (2*(2*((x0 % 2)) + ((-2) + x1)) + 4*x4 + (x0 // 2)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp36, tmp39, tmp40)
    tmp42 = tmp31 >= tmp34
    tmp43 = tl.full([1], 4, tl.int64)
    tmp44 = tmp31 < tmp43
    tmp45 = tmp42 & tmp28
    tmp46 = tl.load(in_ptr6 + (2*((-2) + 2*((x0 % 2)) + ((-2) + x1)) + 4*x4 + (x0 // 2)), tmp45, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (x2), tmp45, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp35, tmp41, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfr4kumx2pdjitchnsorjru6xtst4yl5upevhgt6uga5r3rtmzxy.py
# Topologically Sorted Source Nodes: [out1, out1_1235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1 => add_125, mul_163, mul_164, sub_54
#   out1_1235 => relu_49
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %unsqueeze_433), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_437), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_439), kwargs = {})
#   %relu_49 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_125,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x4 = xindex // 16
    x2 = ((xindex // 16) % 512)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*(x1 // 2) + 8*((x1 % 2)) + 16*x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/wu/cwuh5i744o3toufxxyvbwog4gumnwn4e4il7xjdbpfcvthp6ta23.py
# Topologically Sorted Source Nodes: [out1_1236, out1_1237, out2, out_160], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   out1_1236 => convolution_62
#   out1_1237 => add_127, mul_166, mul_167, sub_55
#   out2 => add_129, mul_169, mul_170, sub_56
#   out_160 => add_130
# Graph fragment:
#   %convolution_62 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_49, %primals_292, %primals_293, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_11, %unsqueeze_449), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_453), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_455), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %add_129), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 16) % 512)
    x3 = (xindex % 4)
    x4 = ((xindex // 4) % 4)
    x6 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3 + 4*(x4 // 2) + 8*((x4 % 2)) + 16*x6), None)
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
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/eo/ceozcvyihpdf2hsvkhvxo3ggj76opmc32zl5wffj6ynlhfea4uqw.py
# Topologically Sorted Source Nodes: [out_161, pad_8], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_161 => relu_50
#   pad_8 => constant_pad_nd_8
# Graph fragment:
#   %relu_50 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_130,), kwargs = {})
#   %constant_pad_nd_8 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_50, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_25 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/25/c25csxgjhvwmpknoxssyv6c5zj3ucox6li7dfg7xy475qugegui7.py
# Topologically Sorted Source Nodes: [out_161, pad_9], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_161 => relu_50
#   pad_9 => constant_pad_nd_9
# Graph fragment:
#   %relu_50 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_130,), kwargs = {})
#   %constant_pad_nd_9 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_50, [1, 1, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_26 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6) % 5)
    x0 = (xindex % 6)
    x2 = xindex // 30
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 4, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp8, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctuyxxn4zxznykqj2b332bjxxntjqrbjcjzhcw5gbsi367esen4s.py
# Topologically Sorted Source Nodes: [out_161, pad_10], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_161 => relu_50
#   pad_10 => constant_pad_nd_10
# Graph fragment:
#   %relu_50 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_130,), kwargs = {})
#   %constant_pad_nd_10 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_50, [1, 0, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_27 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 5) % 6)
    x0 = (xindex % 5)
    x2 = xindex // 30
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp8, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/xc/cxc5kqsy436ombvriynhsmtgey4oyqvxcsobzgew4cd3rtwlwn4x.py
# Topologically Sorted Source Nodes: [out_161, pad_11], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_161 => relu_50
#   pad_11 => constant_pad_nd_11
# Graph fragment:
#   %relu_50 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_130,), kwargs = {})
#   %constant_pad_nd_11 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_50, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_28 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 5) % 5)
    x0 = (xindex % 5)
    x2 = xindex // 25
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2v/c2vwrpxvmeac6ezmoqzrivjahm3bsn53rqqb4v4l2yu5zsfikopq.py
# Topologically Sorted Source Nodes: [stack_8], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_8 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_13, %view_15], 2), kwargs = {})
triton_poi_fused_stack_29 = async_compile.triton('triton_poi_fused_stack_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x4 = xindex // 64
    x2 = ((xindex // 64) % 256)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 4*((x0 % 2)) + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 4, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (4*(4*((x0 % 2)) + (x1)) + 16*x4 + (x0 // 2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 8, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (4*((-4) + 4*((x0 % 2)) + (x1)) + 16*x4 + (x0 // 2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp15, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 8, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = 4*((x0 % 2)) + ((-4) + x1)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 4, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp28
    tmp37 = tl.load(in_ptr4 + (4*(4*((x0 % 2)) + ((-4) + x1)) + 16*x4 + (x0 // 2)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp36, tmp39, tmp40)
    tmp42 = tmp31 >= tmp34
    tmp43 = tl.full([1], 8, tl.int64)
    tmp44 = tmp31 < tmp43
    tmp45 = tmp42 & tmp28
    tmp46 = tl.load(in_ptr6 + (4*((-4) + 4*((x0 % 2)) + ((-4) + x1)) + 16*x4 + (x0 // 2)), tmp45, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (x2), tmp45, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp35, tmp41, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfeszoefdk22cgfpfmf75tsoy6djet3dylytq7ugkbomlbieuf3.py
# Topologically Sorted Source Nodes: [out1_1239, out1_1240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1_1239 => add_132, mul_172, mul_173, sub_57
#   out1_1240 => relu_51
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_17, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_132,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x4 = xindex // 64
    x2 = ((xindex // 64) % 256)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*(x1 // 2) + 32*((x1 % 2)) + 64*x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbo26hqmqaimv3scipp5j2xnc2iwur4rnvkfv4arrrmut6d2f5d.py
# Topologically Sorted Source Nodes: [out1_1241, out1_1242, out2_1236, out_162], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   out1_1241 => convolution_71
#   out1_1242 => add_134, mul_175, mul_176, sub_58
#   out2_1236 => add_136, mul_178, mul_179, sub_59
#   out_162 => add_137
# Graph fragment:
#   %convolution_71 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_51, %primals_322, %primals_323, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_465), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_469), kwargs = {})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_471), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_23, %unsqueeze_473), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_477), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_479), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_134, %add_136), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 64) % 256)
    x3 = (xindex % 8)
    x4 = ((xindex // 8) % 8)
    x6 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3 + 8*(x4 // 2) + 32*((x4 % 2)) + 64*x6), None)
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
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/6e/c6ez2r4aeotgovq564xscgonbmngugtnbsnznoh3r3evf2hcupsz.py
# Topologically Sorted Source Nodes: [out_163, pad_16], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_163 => relu_52
#   pad_16 => constant_pad_nd_16
# Graph fragment:
#   %relu_52 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
#   %constant_pad_nd_16 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_52, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_32 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 10) % 10)
    x0 = (xindex % 10)
    x2 = xindex // 100
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9) + x0 + 8*x1 + 64*x2), tmp10, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/em/ceme3vaifz7i6vlft3ycst5qhhtgszovs4ycixzxrqnd4mnm2mve.py
# Topologically Sorted Source Nodes: [out_163, pad_17], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_163 => relu_52
#   pad_17 => constant_pad_nd_17
# Graph fragment:
#   %relu_52 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
#   %constant_pad_nd_17 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_52, [1, 1, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_33 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 10) % 9)
    x0 = (xindex % 10)
    x2 = xindex // 90
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 8, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-9) + x0 + 8*x1 + 64*x2), tmp8 & xmask, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/26/c26u6lehas5fz43fgoslaul2o2vn557pp3fjtpdi6kal7kq33dpm.py
# Topologically Sorted Source Nodes: [out_163, pad_18], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_163 => relu_52
#   pad_18 => constant_pad_nd_18
# Graph fragment:
#   %relu_52 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
#   %constant_pad_nd_18 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_52, [1, 0, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_34 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 9) % 10)
    x0 = (xindex % 9)
    x2 = xindex // 90
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-9) + x0 + 8*x1 + 64*x2), tmp8 & xmask, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yf/cyfeyqjv6q6shw3kaldjaitrdlmy6tiigwaigap3mqqtjpyg5eai.py
# Topologically Sorted Source Nodes: [out_163, pad_19], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_163 => relu_52
#   pad_19 => constant_pad_nd_19
# Graph fragment:
#   %relu_52 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
#   %constant_pad_nd_19 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_52, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_35 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 9) % 9)
    x0 = (xindex % 9)
    x2 = xindex // 81
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-9) + x0 + 8*x1 + 64*x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/so/csov57m36exzctk3kztkqpqtzq6xvkzwqtitepttps6uyf6l3aa4.py
# Topologically Sorted Source Nodes: [stack_14], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_14 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_25, %view_27], 2), kwargs = {})
triton_poi_fused_stack_36 = async_compile.triton('triton_poi_fused_stack_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x4 = xindex // 256
    x2 = ((xindex // 256) % 128)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 8*((x0 % 2)) + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (8*(8*((x0 % 2)) + (x1)) + 64*x4 + (x0 // 2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (8*((-8) + 8*((x0 % 2)) + (x1)) + 64*x4 + (x0 // 2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp15, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 16, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = 8*((x0 % 2)) + ((-8) + x1)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 8, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp28
    tmp37 = tl.load(in_ptr4 + (8*(8*((x0 % 2)) + ((-8) + x1)) + 64*x4 + (x0 // 2)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp36, tmp39, tmp40)
    tmp42 = tmp31 >= tmp34
    tmp43 = tl.full([1], 16, tl.int64)
    tmp44 = tmp31 < tmp43
    tmp45 = tmp42 & tmp28
    tmp46 = tl.load(in_ptr6 + (8*((-8) + 8*((x0 % 2)) + ((-8) + x1)) + 64*x4 + (x0 // 2)), tmp45, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (x2), tmp45, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp35, tmp41, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/z4/cz463s3cmaqafb4x522wvhbjjf44ik2jk323tjfy55in7qmor3lr.py
# Topologically Sorted Source Nodes: [out1_1244, out1_1245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1_1244 => add_139, mul_181, mul_182, sub_60
#   out1_1245 => relu_53
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_29, %unsqueeze_481), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_485), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_487), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_139,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x4 = xindex // 256
    x2 = ((xindex // 256) % 128)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*(x1 // 2) + 128*((x1 % 2)) + 256*x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/nz/cnzopanezgtfwk2gtycyxipzgws7zz65nbu6aumrfaeozpnmjsz5.py
# Topologically Sorted Source Nodes: [out1_1246, out1_1247, out2_1238, out_164], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   out1_1246 => convolution_80
#   out1_1247 => add_141, mul_184, mul_185, sub_61
#   out2_1238 => add_143, mul_187, mul_188, sub_62
#   out_164 => add_144
# Graph fragment:
#   %convolution_80 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_53, %primals_352, %primals_353, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_35, %unsqueeze_497), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_501), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_503), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %add_143), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 128)
    x3 = (xindex % 16)
    x4 = ((xindex // 16) % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3 + 16*(x4 // 2) + 128*((x4 % 2)) + 256*x6), None)
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
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/fh/cfhxifbu73sqtxyg4kggl2nqxpn3duebxys6jcbsgc4z4q74bid3.py
# Topologically Sorted Source Nodes: [out_165, pad_24], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_165 => relu_54
#   pad_24 => constant_pad_nd_24
# Graph fragment:
#   %relu_54 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_144,), kwargs = {})
#   %constant_pad_nd_24 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_54, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_39 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 18) % 18)
    x0 = (xindex % 18)
    x2 = xindex // 324
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-17) + x0 + 16*x1 + 256*x2), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crsaaduiiim3di7h5wgydt5vbwfhs7vyuoxopfuqcjjmc4dpw6zn.py
# Topologically Sorted Source Nodes: [out_165, pad_25], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_165 => relu_54
#   pad_25 => constant_pad_nd_25
# Graph fragment:
#   %relu_54 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_144,), kwargs = {})
#   %constant_pad_nd_25 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_54, [1, 1, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_40 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 18) % 17)
    x0 = (xindex % 18)
    x2 = xindex // 306
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 16, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-17) + x0 + 16*x1 + 256*x2), tmp8 & xmask, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h7/ch7a56vyesj4aiwobnu2v2dqgjhplhlye4axeouivztbgy7ynev7.py
# Topologically Sorted Source Nodes: [out_165, pad_26], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_165 => relu_54
#   pad_26 => constant_pad_nd_26
# Graph fragment:
#   %relu_54 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_144,), kwargs = {})
#   %constant_pad_nd_26 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_54, [1, 0, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_41 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 17) % 18)
    x0 = (xindex % 17)
    x2 = xindex // 306
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp2 & tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + ((-17) + x0 + 16*x1 + 256*x2), tmp8 & xmask, other=0.0)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yg/cygk2lf5ieov4ateoucal45fzybc6atogjw7eu3yvjmhna2mlagn.py
# Topologically Sorted Source Nodes: [out_165, pad_27], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_165 => relu_54
#   pad_27 => constant_pad_nd_27
# Graph fragment:
#   %relu_54 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_144,), kwargs = {})
#   %constant_pad_nd_27 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_54, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_42 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 17) % 17)
    x0 = (xindex % 17)
    x2 = xindex // 289
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-17) + x0 + 16*x1 + 256*x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fv/cfvkhh2rp4zt6mqi36vhtextrrplg56vp5wyom5dz6w5lhn7xfrf.py
# Topologically Sorted Source Nodes: [stack_20], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_20 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_37, %view_39], 2), kwargs = {})
triton_poi_fused_stack_43 = async_compile.triton('triton_poi_fused_stack_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x4 = xindex // 1024
    x2 = ((xindex // 1024) % 64)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 16*((x0 % 2)) + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 16, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (16*(16*((x0 % 2)) + (x1)) + 256*x4 + (x0 // 2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 32, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (16*((-16) + 16*((x0 % 2)) + (x1)) + 256*x4 + (x0 // 2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp15, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 32, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = 16*((x0 % 2)) + ((-16) + x1)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 16, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp28
    tmp37 = tl.load(in_ptr4 + (16*(16*((x0 % 2)) + ((-16) + x1)) + 256*x4 + (x0 // 2)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp36, tmp39, tmp40)
    tmp42 = tmp31 >= tmp34
    tmp43 = tl.full([1], 32, tl.int64)
    tmp44 = tmp31 < tmp43
    tmp45 = tmp42 & tmp28
    tmp46 = tl.load(in_ptr6 + (16*((-16) + 16*((x0 % 2)) + ((-16) + x1)) + 256*x4 + (x0 // 2)), tmp45, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (x2), tmp45, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp35, tmp41, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjbm3zq7duemitszwuy5l223f7kfs2pr24a272py6ytvx4vja3x.py
# Topologically Sorted Source Nodes: [out1_1249, out1_1250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1_1249 => add_146, mul_190, mul_191, sub_63
#   out1_1250 => relu_55
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_41, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_146,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x4 = xindex // 1024
    x2 = ((xindex // 1024) % 64)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 2) + 512*((x1 % 2)) + 1024*x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zn/cznhww64bupyooppjgpbs2z7zwnkyoxtc37wuxem2zreoistyn2d.py
# Topologically Sorted Source Nodes: [out1_1251, out1_1252, out2_1240, out_166, out_167], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out1_1251 => convolution_89
#   out1_1252 => add_148, mul_193, mul_194, sub_64
#   out2_1240 => add_150, mul_196, mul_197, sub_65
#   out_166 => add_151
#   out_167 => relu_56
# Graph fragment:
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_55, %primals_382, %primals_383, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_47, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_148, %add_150), kwargs = {})
#   %relu_56 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_45', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_45(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 1024) % 64)
    x3 = (xindex % 32)
    x4 = ((xindex // 32) % 32)
    x6 = xindex // 1024
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3 + 32*(x4 // 2) + 512*((x4 % 2)) + 1024*x6), None)
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
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(in_out_ptr1 + (x5), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/ma/cmacggwn2mueiywttjtm7cxgc32z3uxaduvs5cqpv4h3y6ru3w3n.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_9 => convert_element_type_133
# Graph fragment:
#   %convert_element_type_133 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_48, torch.int64), kwargs = {})
triton_poi_fused__to_copy_46 = async_compile.triton('triton_poi_fused__to_copy_46', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.14035087719298245
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4hq2hrikmjlt65avydstsfsrhx7bwdh3sbdwmnuh6k34cdmggxm.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_9 => add_153, clamp_max
# Graph fragment:
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_133, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_153, 31), kwargs = {})
triton_poi_fused_add_clamp_47 = async_compile.triton('triton_poi_fused_add_clamp_47', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.14035087719298245
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cd/ccd7tqvkhrdgb3pgxojiivx77r35xmmegej4amh7fuvunnkn5t3y.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_9 => add_154, clamp_min_1, convert_element_type_134, convert_element_type_135, iota_1, mul_199, sub_67
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (304,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_134, 0.5), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, 0.10526315789473684), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_199, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_67, 0.0), kwargs = {})
#   %convert_element_type_135 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.10526315789473684
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjobpldsvxukcfjweqg4fp44kr7xnc5dpgyiuwpbzvwvm73uddyx.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_9 => add_155, clamp_max_1
# Graph fragment:
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_135, 1), kwargs = {})
#   %clamp_max_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_155, 31), kwargs = {})
triton_poi_fused_add_clamp_49 = async_compile.triton('triton_poi_fused_add_clamp_49', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_49(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.10526315789473684
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zf/czfz6cnibqe22ceizcwcd4tdkjrygfe6bv4jshmu3zmppkm6wgad.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_9 => add_154, clamp_max_2, clamp_min_1, clamp_min_2, convert_element_type_134, iota_1, mul_199, sub_67, sub_68
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (304,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_134, 0.5), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, 0.10526315789473684), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_199, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_67, 0.0), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_135), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_68, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_50 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_50', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_50(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.10526315789473684
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bn/cbnvevpiddrdhu4uuzy7ty7pnbzf2bbo65u6jhsewz4xqnqkbn22.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_9 => clamp_max_3, clamp_min_3, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_48, %convert_element_type_133), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_71, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
triton_poi_fused_clamp_sub_51 = async_compile.triton('triton_poi_fused_clamp_sub_51', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_sub_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_sub_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.14035087719298245
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yh/cyhl4wldfibsyk5ychdlbmhngxn7jvc5vpq5gm435xlxetcdtp4d.py
# Topologically Sorted Source Nodes: [x_7, x_8, x_9], Original ATen: [aten.convolution, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_7 => convolution_90
#   x_8 => relu_57
#   x_9 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_156, add_157, add_158, mul_200, mul_201, mul_202, sub_69, sub_70, sub_72
# Graph fragment:
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %primals_392, %primals_393, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_57 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_57, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_57, [None, None, %convert_element_type_133, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_57, [None, None, %clamp_max, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_57, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %clamp_max_2), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_200), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %clamp_max_2), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_201), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_157, %add_156), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %clamp_max_3), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %mul_202), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_52 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_52', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 277248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 304) % 228)
    x0 = (xindex % 304)
    x2 = xindex // 69312
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr2 + (tmp18 + 32*tmp4 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp11
    tmp21 = triton_helpers.maximum(tmp13, tmp20)
    tmp22 = tmp21 - tmp14
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 + tmp24
    tmp27 = tmp26 + tmp1
    tmp28 = tmp26 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp26)
    tmp30 = tl.load(in_ptr2 + (tmp8 + 32*tmp29 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp11
    tmp32 = triton_helpers.maximum(tmp13, tmp31)
    tmp33 = tl.load(in_ptr2 + (tmp18 + 32*tmp29 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 + tmp11
    tmp35 = triton_helpers.maximum(tmp13, tmp34)
    tmp36 = tmp35 - tmp32
    tmp37 = tmp36 * tmp23
    tmp38 = tmp32 + tmp37
    tmp39 = tmp38 - tmp25
    tmp41 = tmp39 * tmp40
    tmp42 = tmp25 + tmp41
    tl.store(in_out_ptr0 + (x3), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbeps6wrkeymaxyk2mpj5n6xigdm4wqae5ddbebsbbt6my6eanr.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_7 => convolution_90
#   x_8 => relu_57
# Graph fragment:
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_56, %primals_392, %primals_393, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_57 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_90,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_57, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_53 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_53(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.0
    tmp7 = tmp5 <= tmp6
    tl.store(out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393 = args
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
    assert_size_stride(primals_122, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_137, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_138, (1024, ), (1, ))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, ), (1, ))
    assert_size_stride(primals_152, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, ), (1, ))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (256, ), (1, ))
    assert_size_stride(primals_162, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (256, ), (1, ))
    assert_size_stride(primals_165, (256, ), (1, ))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_167, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_181, (256, ), (1, ))
    assert_size_stride(primals_182, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_187, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (1024, 256, 1, 1), (256, 1, 1, 1))
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
    assert_size_stride(primals_217, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_218, (512, ), (1, ))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_222, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_228, (2048, ), (1, ))
    assert_size_stride(primals_229, (2048, ), (1, ))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_231, (2048, ), (1, ))
    assert_size_stride(primals_232, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_233, (2048, ), (1, ))
    assert_size_stride(primals_234, (2048, ), (1, ))
    assert_size_stride(primals_235, (2048, ), (1, ))
    assert_size_stride(primals_236, (2048, ), (1, ))
    assert_size_stride(primals_237, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (512, ), (1, ))
    assert_size_stride(primals_242, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, ), (1, ))
    assert_size_stride(primals_251, (2048, ), (1, ))
    assert_size_stride(primals_252, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_260, (512, ), (1, ))
    assert_size_stride(primals_261, (512, ), (1, ))
    assert_size_stride(primals_262, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_263, (2048, ), (1, ))
    assert_size_stride(primals_264, (2048, ), (1, ))
    assert_size_stride(primals_265, (2048, ), (1, ))
    assert_size_stride(primals_266, (2048, ), (1, ))
    assert_size_stride(primals_267, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_268, (1024, ), (1, ))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, 1024, 2, 3), (6144, 6, 3, 1))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, 1024, 3, 2), (6144, 6, 2, 1))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_278, (512, 1024, 2, 2), (4096, 4, 2, 1))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_281, (512, ), (1, ))
    assert_size_stride(primals_282, (512, 1024, 2, 3), (6144, 6, 3, 1))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (512, 1024, 3, 2), (6144, 6, 2, 1))
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (512, 1024, 2, 2), (4096, 4, 2, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (512, ), (1, ))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (512, ), (1, ))
    assert_size_stride(primals_302, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_304, (256, 512, 2, 3), (3072, 6, 3, 1))
    assert_size_stride(primals_305, (256, ), (1, ))
    assert_size_stride(primals_306, (256, 512, 3, 2), (3072, 6, 2, 1))
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (256, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(primals_309, (256, ), (1, ))
    assert_size_stride(primals_310, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_311, (256, ), (1, ))
    assert_size_stride(primals_312, (256, 512, 2, 3), (3072, 6, 3, 1))
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, 512, 3, 2), (3072, 6, 2, 1))
    assert_size_stride(primals_315, (256, ), (1, ))
    assert_size_stride(primals_316, (256, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(primals_317, (256, ), (1, ))
    assert_size_stride(primals_318, (256, ), (1, ))
    assert_size_stride(primals_319, (256, ), (1, ))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (256, ), (1, ))
    assert_size_stride(primals_322, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_323, (256, ), (1, ))
    assert_size_stride(primals_324, (256, ), (1, ))
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (256, ), (1, ))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (256, ), (1, ))
    assert_size_stride(primals_330, (256, ), (1, ))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, 256, 2, 3), (1536, 6, 3, 1))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (128, 256, 3, 2), (1536, 6, 2, 1))
    assert_size_stride(primals_337, (128, ), (1, ))
    assert_size_stride(primals_338, (128, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(primals_339, (128, ), (1, ))
    assert_size_stride(primals_340, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_342, (128, 256, 2, 3), (1536, 6, 3, 1))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (128, 256, 3, 2), (1536, 6, 2, 1))
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (128, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (128, ), (1, ))
    assert_size_stride(primals_351, (128, ), (1, ))
    assert_size_stride(primals_352, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_353, (128, ), (1, ))
    assert_size_stride(primals_354, (128, ), (1, ))
    assert_size_stride(primals_355, (128, ), (1, ))
    assert_size_stride(primals_356, (128, ), (1, ))
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (128, ), (1, ))
    assert_size_stride(primals_360, (128, ), (1, ))
    assert_size_stride(primals_361, (128, ), (1, ))
    assert_size_stride(primals_362, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, 128, 2, 3), (768, 6, 3, 1))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, 128, 3, 2), (768, 6, 2, 1))
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (64, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_371, (64, ), (1, ))
    assert_size_stride(primals_372, (64, 128, 2, 3), (768, 6, 3, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, 128, 3, 2), (768, 6, 2, 1))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, ), (1, ))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (64, ), (1, ))
    assert_size_stride(primals_381, (64, ), (1, ))
    assert_size_stride(primals_382, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (64, ), (1, ))
    assert_size_stride(primals_385, (64, ), (1, ))
    assert_size_stride(primals_386, (64, ), (1, ))
    assert_size_stride(primals_387, (64, ), (1, ))
    assert_size_stride(primals_388, (64, ), (1, ))
    assert_size_stride(primals_389, (64, ), (1, ))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (1, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_393, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        buf2 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
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
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf2, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf10 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
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
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf23, primals_72, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf30 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
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
        assert_size_stride(buf50, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf51 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf50, primals_123, primals_124, primals_125, primals_126, buf51, 65536, grid=grid(65536), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_127, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf53 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf52, primals_128, primals_129, primals_130, primals_131, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 1024, 4, 4), (16384, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf49, primals_137, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf56 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf57, buf54, primals_133, primals_134, primals_135, primals_136, buf55, primals_138, primals_139, primals_140, primals_141, 65536, grid=grid(65536), stream=stream0)
        del primals_136
        del primals_141
        # Topologically Sorted Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf58, primals_143, primals_144, primals_145, primals_146, buf59, 16384, grid=grid(16384), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf61 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf60, primals_148, primals_149, primals_150, primals_151, buf61, 16384, grid=grid(16384), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf63 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_87, out_88, out_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf62, primals_153, primals_154, primals_155, primals_156, buf57, buf63, 65536, grid=grid(65536), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf65 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf64, primals_158, primals_159, primals_160, primals_161, buf65, 16384, grid=grid(16384), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf67 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf66, primals_163, primals_164, primals_165, primals_166, buf67, 16384, grid=grid(16384), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf69 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf68, primals_168, primals_169, primals_170, primals_171, buf63, buf69, 65536, grid=grid(65536), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf71 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf70, primals_173, primals_174, primals_175, primals_176, buf71, 16384, grid=grid(16384), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf73 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf72, primals_178, primals_179, primals_180, primals_181, buf73, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf75 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf74, primals_183, primals_184, primals_185, primals_186, buf69, buf75, 65536, grid=grid(65536), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf77 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf76, primals_188, primals_189, primals_190, primals_191, buf77, 16384, grid=grid(16384), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf79 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf78, primals_193, primals_194, primals_195, primals_196, buf79, 16384, grid=grid(16384), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf81 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_117, out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf80, primals_198, primals_199, primals_200, primals_201, buf75, buf81, 65536, grid=grid(65536), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf83 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf82, primals_203, primals_204, primals_205, primals_206, buf83, 16384, grid=grid(16384), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf85 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf84, primals_208, primals_209, primals_210, primals_211, buf85, 16384, grid=grid(16384), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf87 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_127, out_128, out_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf86, primals_213, primals_214, primals_215, primals_216, buf81, buf87, 65536, grid=grid(65536), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf89 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf88, primals_218, primals_219, primals_220, primals_221, buf89, 32768, grid=grid(32768), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_222, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf91 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf90, primals_223, primals_224, primals_225, primals_226, buf91, 8192, grid=grid(8192), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 2048, 2, 2), (8192, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf87, primals_232, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf94 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf95, buf92, primals_228, primals_229, primals_230, primals_231, buf93, primals_233, primals_234, primals_235, primals_236, 32768, grid=grid(32768), stream=stream0)
        del primals_231
        del primals_236
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf97 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf96, primals_238, primals_239, primals_240, primals_241, buf97, 8192, grid=grid(8192), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf99 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf98, primals_243, primals_244, primals_245, primals_246, buf99, 8192, grid=grid(8192), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf101 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf100, primals_248, primals_249, primals_250, primals_251, buf95, buf101, 32768, grid=grid(32768), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf103 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf102, primals_253, primals_254, primals_255, primals_256, buf103, 8192, grid=grid(8192), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf105 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf104, primals_258, primals_259, primals_260, primals_261, buf105, 8192, grid=grid(8192), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf107 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_157, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf106, primals_263, primals_264, primals_265, primals_266, buf101, buf107, 32768, grid=grid(32768), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf109 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf108, primals_268, primals_269, primals_270, primals_271, buf109, 16384, grid=grid(16384), stream=stream0)
        del primals_271
        buf110 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_18.run(buf109, buf110, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_1], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf112 = empty_strided_cuda((4, 1024, 3, 4), (12288, 12, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_19.run(buf109, buf112, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_2], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf114 = empty_strided_cuda((4, 1024, 4, 3), (12288, 12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_20.run(buf109, buf114, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_3], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf116 = empty_strided_cuda((4, 1024, 3, 3), (9216, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_21.run(buf109, buf116, 36864, grid=grid(36864), stream=stream0)
        del buf109
        # Topologically Sorted Source Nodes: [out1_4], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [out2_1], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf110, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [out2_2], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf112, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [out2_3], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf114, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [out2_4], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf116, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf122 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_2], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_22.run(buf111, primals_273, buf113, primals_275, buf115, primals_277, buf117, primals_279, buf122, 32768, grid=grid(32768), stream=stream0)
        del buf111
        del buf113
        del buf115
        del buf117
        del primals_273
        del primals_275
        del primals_277
        del primals_279
        buf123 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_5], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_22.run(buf118, primals_281, buf119, primals_283, buf120, primals_285, buf121, primals_287, buf123, 32768, grid=grid(32768), stream=stream0)
        del buf118
        del buf119
        del buf120
        del buf121
        del primals_281
        del primals_283
        del primals_285
        del primals_287
        buf124 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1, out1_1235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf122, primals_288, primals_289, primals_290, primals_291, buf124, 32768, grid=grid(32768), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [out1_1236], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf126 = buf125; del buf125  # reuse
        buf127 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1236, out1_1237, out2, out_160], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_24.run(buf126, primals_293, primals_294, primals_295, primals_296, primals_297, buf123, primals_298, primals_299, primals_300, primals_301, buf127, 32768, grid=grid(32768), stream=stream0)
        del primals_293
        buf128 = empty_strided_cuda((4, 512, 6, 6), (18432, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_161, pad_8], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_25.run(buf127, buf128, 73728, grid=grid(73728), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_5], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf130 = empty_strided_cuda((4, 512, 5, 6), (15360, 30, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_161, pad_9], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_26.run(buf127, buf130, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_6], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf132 = empty_strided_cuda((4, 512, 6, 5), (15360, 30, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_161, pad_10], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_27.run(buf127, buf132, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_7], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf134 = empty_strided_cuda((4, 512, 5, 5), (12800, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_161, pad_11], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_28.run(buf127, buf134, 51200, grid=grid(51200), stream=stream0)
        del buf127
        # Topologically Sorted Source Nodes: [out1_8], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out2_5], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf128, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out2_6], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf130, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out2_7], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf132, primals_314, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 256, 4, 4), (4096, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out2_8], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf134, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf140 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_8], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_29.run(buf129, primals_303, buf131, primals_305, buf133, primals_307, buf135, primals_309, buf140, 65536, grid=grid(65536), stream=stream0)
        del buf129
        del buf131
        del buf133
        del buf135
        del primals_303
        del primals_305
        del primals_307
        del primals_309
        buf141 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_11], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_29.run(buf136, primals_311, buf137, primals_313, buf138, primals_315, buf139, primals_317, buf141, 65536, grid=grid(65536), stream=stream0)
        del buf136
        del buf137
        del buf138
        del buf139
        del primals_311
        del primals_313
        del primals_315
        del primals_317
        buf142 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1239, out1_1240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf140, primals_318, primals_319, primals_320, primals_321, buf142, 65536, grid=grid(65536), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [out1_1241], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf144 = buf143; del buf143  # reuse
        buf145 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1241, out1_1242, out2_1236, out_162], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_31.run(buf144, primals_323, primals_324, primals_325, primals_326, primals_327, buf141, primals_328, primals_329, primals_330, primals_331, buf145, 65536, grid=grid(65536), stream=stream0)
        del primals_323
        buf146 = empty_strided_cuda((4, 256, 10, 10), (25600, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, pad_16], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_32.run(buf145, buf146, 102400, grid=grid(102400), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_9], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf148 = empty_strided_cuda((4, 256, 9, 10), (23040, 90, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, pad_17], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_33.run(buf145, buf148, 92160, grid=grid(92160), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_10], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf150 = empty_strided_cuda((4, 256, 10, 9), (23040, 90, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, pad_18], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_34.run(buf145, buf150, 92160, grid=grid(92160), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_11], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf152 = empty_strided_cuda((4, 256, 9, 9), (20736, 81, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, pad_19], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_35.run(buf145, buf152, 82944, grid=grid(82944), stream=stream0)
        del buf145
        # Topologically Sorted Source Nodes: [out1_12], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out2_9], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf146, primals_340, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out2_10], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf148, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out2_11], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf150, primals_344, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 128, 8, 8), (8192, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out2_12], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf152, primals_346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf158 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_14], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_36.run(buf147, primals_333, buf149, primals_335, buf151, primals_337, buf153, primals_339, buf158, 131072, grid=grid(131072), stream=stream0)
        del buf147
        del buf149
        del buf151
        del buf153
        del primals_333
        del primals_335
        del primals_337
        del primals_339
        buf159 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_17], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_36.run(buf154, primals_341, buf155, primals_343, buf156, primals_345, buf157, primals_347, buf159, 131072, grid=grid(131072), stream=stream0)
        del buf154
        del buf155
        del buf156
        del buf157
        del primals_341
        del primals_343
        del primals_345
        del primals_347
        buf160 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1244, out1_1245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf158, primals_348, primals_349, primals_350, primals_351, buf160, 131072, grid=grid(131072), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [out1_1246], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1246, out1_1247, out2_1238, out_164], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_38.run(buf162, primals_353, primals_354, primals_355, primals_356, primals_357, buf159, primals_358, primals_359, primals_360, primals_361, buf163, 131072, grid=grid(131072), stream=stream0)
        del primals_353
        buf164 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_165, pad_24], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_39.run(buf163, buf164, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_13], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf166 = empty_strided_cuda((4, 128, 17, 18), (39168, 306, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_165, pad_25], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_40.run(buf163, buf166, 156672, grid=grid(156672), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_14], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_364, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf168 = empty_strided_cuda((4, 128, 18, 17), (39168, 306, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_165, pad_26], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_41.run(buf163, buf168, 156672, grid=grid(156672), stream=stream0)
        # Topologically Sorted Source Nodes: [out1_15], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf170 = empty_strided_cuda((4, 128, 17, 17), (36992, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_165, pad_27], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_42.run(buf163, buf170, 147968, grid=grid(147968), stream=stream0)
        del buf163
        # Topologically Sorted Source Nodes: [out1_16], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_368, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out2_13], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf164, primals_370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out2_14], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf166, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out2_15], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf168, primals_374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out2_16], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf170, primals_376, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf176 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_20], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_43.run(buf165, primals_363, buf167, primals_365, buf169, primals_367, buf171, primals_369, buf176, 262144, grid=grid(262144), stream=stream0)
        del buf165
        del buf167
        del buf169
        del buf171
        del primals_363
        del primals_365
        del primals_367
        del primals_369
        buf177 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_23], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_43.run(buf172, primals_371, buf173, primals_373, buf174, primals_375, buf175, primals_377, buf177, 262144, grid=grid(262144), stream=stream0)
        del buf172
        del buf173
        del buf174
        del buf175
        del primals_371
        del primals_373
        del primals_375
        del primals_377
        buf178 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_1249, out1_1250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf176, primals_378, primals_379, primals_380, primals_381, buf178, 262144, grid=grid(262144), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [out1_1251], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf180 = buf179; del buf179  # reuse
        buf181 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [out1_1251, out1_1252, out2_1240, out_166, out_167], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_45.run(buf180, buf182, primals_383, primals_384, primals_385, primals_386, primals_387, buf177, primals_388, primals_389, primals_390, primals_391, 262144, grid=grid(262144), stream=stream0)
        del primals_383
        del primals_387
        del primals_391
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf184 = empty_strided_cuda((228, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_46.run(buf184, 228, grid=grid(228), stream=stream0)
        buf185 = empty_strided_cuda((228, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_47.run(buf185, 228, grid=grid(228), stream=stream0)
        buf186 = empty_strided_cuda((304, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48.run(buf186, 304, grid=grid(304), stream=stream0)
        buf187 = empty_strided_cuda((304, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_49.run(buf187, 304, grid=grid(304), stream=stream0)
        buf188 = empty_strided_cuda((304, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_50.run(buf188, 304, grid=grid(304), stream=stream0)
        buf190 = empty_strided_cuda((228, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_sub_51.run(buf190, 228, grid=grid(228), stream=stream0)
        buf189 = empty_strided_cuda((4, 1, 228, 304), (69312, 277248, 304, 1), torch.float32)
        buf192 = reinterpret_tensor(buf189, (4, 1, 228, 304), (69312, 69312, 304, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8, x_9], Original ATen: [aten.convolution, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_52.run(buf192, buf184, buf186, buf183, primals_393, buf187, buf188, buf185, buf190, 277248, grid=grid(277248), stream=stream0)
        buf193 = empty_strided_cuda((4, 1, 32, 32), (1024, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_53.run(buf183, primals_393, buf193, 4096, grid=grid(4096), stream=stream0)
        del buf183
        del primals_393
    return (buf192, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_274, primals_276, primals_278, primals_280, primals_282, primals_284, primals_286, primals_288, primals_289, primals_290, primals_292, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_308, primals_310, primals_312, primals_314, primals_316, primals_318, primals_319, primals_320, primals_322, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_334, primals_336, primals_338, primals_340, primals_342, primals_344, primals_346, primals_348, primals_349, primals_350, primals_352, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_364, primals_366, primals_368, primals_370, primals_372, primals_374, primals_376, primals_378, primals_379, primals_380, primals_382, primals_384, primals_385, primals_386, primals_388, primals_389, primals_390, primals_392, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf110, buf112, buf114, buf116, buf122, buf123, buf124, buf126, buf128, buf130, buf132, buf134, buf140, buf141, buf142, buf144, buf146, buf148, buf150, buf152, buf158, buf159, buf160, buf162, buf164, buf166, buf168, buf170, buf176, buf177, buf178, buf180, buf182, buf184, buf185, buf186, buf187, buf188, buf190, buf193, )


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
    primals_122 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_217 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, 1024, 2, 3), (6144, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, 1024, 3, 2), (6144, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, 1024, 2, 2), (4096, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, 1024, 2, 3), (6144, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, 1024, 3, 2), (6144, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, 1024, 2, 2), (4096, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, 512, 2, 3), (3072, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((256, 512, 3, 2), (3072, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((256, 512, 2, 3), (3072, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((256, 512, 3, 2), (3072, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((256, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, 256, 2, 3), (1536, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((128, 256, 3, 2), (1536, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((128, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, 256, 2, 3), (1536, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((128, 256, 3, 2), (1536, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, 128, 2, 3), (768, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, 128, 3, 2), (768, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 128, 2, 3), (768, 6, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, 128, 3, 2), (768, 6, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
