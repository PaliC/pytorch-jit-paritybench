# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/si/csivq5uqi2o6lyi54xqj6j5apugbb2wvhz2zi4rz6horhbwlpfsh.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_4, mul_5, sub_1
#   x_5 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jt/cjt7bkcwqcfykdchkhvjhbg3qyzotgt62wkr5zg3wrsafb4d6w7n.py
# Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_11, mul_16, mul_17, sub_5
#   out_7 => add_9, mul_13, mul_14, sub_4
#   out_8 => add_12
#   out_9 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %add_11), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/w5/cw5fupiwswa6kginduaijqy3iffgmp4ebd3cx2edaf35hthyqajc.py
# Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_17 => add_18, mul_25, mul_26, sub_8
#   out_18 => add_19
#   out_19 => relu_7
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %relu_4), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6m/c6msl2nkwn2wxdrcjzwys4zkjbawmj3pdxiwgvqqjcq5azqj4wtl.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_35, mul_46, mul_47, sub_15
#   input_5 => relu_14
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
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
    x1 = ((xindex // 256) % 4)
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


# kernel path: inductor_cache/mg/cmgefaka62udrqw5tbvirmzovqibawn63zyulhbubpgqmgdhlswk.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_7 => add_37, mul_49, mul_50, sub_16
#   input_8 => relu_15
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_15 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 8)
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


# kernel path: inductor_cache/gq/cgqkwit3n2ihcy4berbweb4oclpru53ugin47mx4dz2sbzspglkt.py
# Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_44 => add_41, mul_55, mul_56, sub_18
#   out_45 => add_42
#   out_46 => relu_17
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %relu_14), kwargs = {})
#   %relu_17 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
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


# kernel path: inductor_cache/76/c76xjtl2kfrdjoqmas6smexsgfs5aijj57ztgoynr5c23l4gukzu.py
# Topologically Sorted Source Nodes: [out_72, out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_72 => add_61, mul_79, mul_80, sub_26
#   out_73 => add_62
#   out_74 => relu_25
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %relu_15), kwargs = {})
#   %relu_25 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_62,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4i6nlbsxdarqopnyh3rjrmp62vioezer2el425p4qcdxvyqm45g.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_10 => add_79, mul_100, mul_101, sub_33
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_79 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ts/ctsye25bdgbo4qp3e6foslhvhk7hmo6wr6numzunhgatut4qs7vj.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate => convert_element_type_69
# Graph fragment:
#   %convert_element_type_69 : [num_users=21] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdozpojsnhw7tvsuvdri2b7mad2lfp43om5jhcronoair57klqho.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate => add_80, clamp_max
# Graph fragment:
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_69, 1), kwargs = {})
#   %clamp_max : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_80, 7), kwargs = {})
triton_poi_fused_add_clamp_10 = async_compile.triton('triton_poi_fused_add_clamp_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 7, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bf/cbfnwlkty2phpmvuutzbc2jwvmpdd2mfrphzrxc4ag2c77xilhoz.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate => clamp_max_2, clamp_min, clamp_min_2, convert_element_type_68, iota, mul_102, sub_34
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_68 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, 0.4666666666666667), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_102, 0.0), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_71), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_34, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_11 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/con6maqh5i7vomnp5tubyuryx5wqb53xkbwmeq4d3pavi7ikmslt.py
# Topologically Sorted Source Nodes: [out_65, out_66, out_67, interpolate, y, residual], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_82, add_83, add_84, mul_104, mul_105, mul_106, sub_35, sub_36, sub_38
#   out_65 => add_56, mul_73, mul_74, sub_24
#   out_66 => add_57
#   out_67 => relu_23
#   residual => relu_32
#   y => add_85
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %relu_21), kwargs = {})
#   %relu_23 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_79, [None, None, %convert_element_type_69, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_79, [None, None, %convert_element_type_69, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_79, [None, None, %clamp_max, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_79, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %clamp_max_2), kwargs = {})
#   %add_82 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_104), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_2), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_105), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_83, %add_82), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %clamp_max_3), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %mul_106), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_23, %add_84), kwargs = {})
#   %relu_32 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x6 = xindex
    x1 = ((xindex // 256) % 4)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp40 = tmp39 + tmp21
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr8 + (tmp28 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (tmp33 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45 * tmp36
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47 - tmp38
    tmp50 = tmp48 * tmp49
    tmp51 = tmp38 + tmp50
    tmp52 = tmp19 + tmp51
    tmp53 = triton_helpers.maximum(tmp18, tmp52)
    tl.store(out_ptr0 + (x6), tmp19, None)
    tl.store(in_out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/36/c36273zkz27gpcm72mnzpfh3cqk7qzyu7p2n3xay4ctct5n3maxb.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_90, mul_111, mul_112, sub_40
#   input_15 => relu_34
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_283), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, %unsqueeze_285), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %unsqueeze_287), kwargs = {})
#   %relu_34 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_90,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
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


# kernel path: inductor_cache/kv/ckv6fqbyn22lglc7dlzcdeuddjuqnk7kpr2icq42iiitmhfsvjnn.py
# Topologically Sorted Source Nodes: [out_156, out_157, out_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_156 => add_134, mul_165, mul_166, sub_58
#   out_157 => add_135
#   out_158 => relu_52
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_427), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %unsqueeze_429), kwargs = {})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %unsqueeze_431), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_134, %relu_34), kwargs = {})
#   %relu_52 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t3/ct3xkbrscrdlqxs5jzmu433xdlxqjw6iqfzc3oph6erbzmiucilp.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_19 => add_160, mul_194, mul_195, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_491), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_194, %unsqueeze_493), kwargs = {})
#   %add_160 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %unsqueeze_495), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pw/cpw5dgo3a6ee6pnbcswde22pcqtfcwngmr46kiwbkwtgvajr6uvq.py
# Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_2 => convert_element_type_133
# Graph fragment:
#   %convert_element_type_133 : [num_users=19] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.2
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5p/c5po2tqqr3aqxtubfz4b7mprtkekw4hnszj2lnxum2ufr3j23rdy.py
# Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_2 => add_161, clamp_max_8
# Graph fragment:
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_133, 1), kwargs = {})
#   %clamp_max_8 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_161, 3), kwargs = {})
triton_poi_fused_add_clamp_17 = async_compile.triton('triton_poi_fused_add_clamp_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.2
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hn/chnnfommjzq3jq352ndjnq57ijnxkk547ykckjmh5srzdrsvuwrn.py
# Topologically Sorted Source Nodes: [interpolate, interpolate_2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate => convert_element_type_68, iota
#   interpolate_2 => clamp_max_10, clamp_min_10, clamp_min_8, mul_196, sub_72
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_68 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, 0.2), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_196, 0.0), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_135), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_72, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_18 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_18(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.2
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hu/chu5qexfywxp23amcnlptldolpcydjzacikdtg2lecumzj3z7yid.py
# Topologically Sorted Source Nodes: [out_121, out_122, out_123, interpolate_1, y_2, interpolate_2, y_3, residual_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   interpolate_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_155, add_156, add_157, mul_190, mul_191, mul_192, sub_67, sub_68, sub_70
#   interpolate_2 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_163, add_164, add_165, mul_198, mul_199, mul_200, sub_73, sub_74, sub_76
#   out_121 => add_109, mul_135, mul_136, sub_48
#   out_122 => add_110
#   out_123 => relu_42
#   residual_2 => relu_59
#   y_2 => add_158
#   y_3 => add_166
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_347), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_135, %unsqueeze_349), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, %unsqueeze_351), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %relu_40), kwargs = {})
#   %relu_42 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_110,), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_152, [None, None, %convert_element_type_69, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_152, [None, None, %convert_element_type_69, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_152, [None, None, %clamp_max, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_152, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %clamp_max_2), kwargs = {})
#   %add_155 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_190), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %clamp_max_2), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_191), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_156, %add_155), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %clamp_max_3), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_155, %mul_192), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_42, %add_157), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_160, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_160, [None, None, %convert_element_type_133, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_160, [None, None, %clamp_max_8, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_160, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %clamp_max_10), kwargs = {})
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_198), kwargs = {})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %clamp_max_10), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_199), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_164, %add_163), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %clamp_max_11), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %mul_200), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %add_165), kwargs = {})
#   %relu_59 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_166,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x6 = xindex
    x1 = ((xindex // 256) % 4)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x3), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x3), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr17 + (x3), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp40 = tmp39 + tmp21
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr8 + (tmp28 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (tmp33 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45 * tmp36
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47 - tmp38
    tmp50 = tmp48 * tmp49
    tmp51 = tmp38 + tmp50
    tmp52 = tmp19 + tmp51
    tmp54 = tl.full([XBLOCK], 4, tl.int32)
    tmp55 = tmp53 + tmp54
    tmp56 = tmp53 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp53)
    tmp59 = tmp58 + tmp54
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tmp62 = tl.load(in_ptr15 + (tmp61 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp64 = tmp63 + tmp54
    tmp65 = tmp63 < 0
    tmp66 = tl.where(tmp65, tmp64, tmp63)
    tmp67 = tl.load(in_ptr15 + (tmp66 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp68 = tmp67 - tmp62
    tmp70 = tmp68 * tmp69
    tmp71 = tmp62 + tmp70
    tmp73 = tmp72 + tmp54
    tmp74 = tmp72 < 0
    tmp75 = tl.where(tmp74, tmp73, tmp72)
    tmp76 = tl.load(in_ptr15 + (tmp61 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr15 + (tmp66 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp78 = tmp77 - tmp76
    tmp79 = tmp78 * tmp69
    tmp80 = tmp76 + tmp79
    tmp81 = tmp80 - tmp71
    tmp83 = tmp81 * tmp82
    tmp84 = tmp71 + tmp83
    tmp85 = tmp52 + tmp84
    tmp86 = triton_helpers.maximum(tmp18, tmp85)
    tl.store(out_ptr0 + (x6), tmp19, None)
    tl.store(in_out_ptr0 + (x6), tmp86, None)
''', device_str='cuda')


# kernel path: inductor_cache/3i/c3i2ooxe2bnejvf36xf2ln7nhboiu2t3r5zxfvq7pjuafl7vopvf.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_23 => add_171, mul_205, mul_206, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_507), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_509), kwargs = {})
#   %add_171 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_511), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 8)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpd7lxikc77jbqgdggqmw2p5b3ycek6v65gl5lr3ufnkkqki5hxe.py
# Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_3 => convert_element_type_141
# Graph fragment:
#   %convert_element_type_141 : [num_users=17] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x4/cx4z6dm32rmsezqmxunonjom7vnkiw4hgjebiqgaaznjiyia3uz4.py
# Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_3 => add_172, clamp_max_12
# Graph fragment:
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_141, 1), kwargs = {})
#   %clamp_max_12 : [num_users=15] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_172, 3), kwargs = {})
triton_poi_fused_add_clamp_22 = async_compile.triton('triton_poi_fused_add_clamp_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd7yahicukgb4btsntcmjgnwmnyg5rrdcedjmccqyb2eyrwggibo.py
# Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate_3 => clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_140, iota_6, mul_207, sub_79
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_140 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_140, 0.42857142857142855), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_207, 0.0), kwargs = {})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_143), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_79, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=15] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_23 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kr/ckrmm25vyawrbds63ecfdkyqhiraf64uh3xmddw5gtxs2b66ng6b.py
# Topologically Sorted Source Nodes: [input_21, y_4, interpolate_3, y_5, residual_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
# Source node to ATen node mapping:
#   input_21 => add_168, mul_202, mul_203, sub_77
#   interpolate_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_174, add_175, add_176, mul_209, mul_210, mul_211, sub_80, sub_81, sub_83
#   residual_3 => relu_60
#   y_4 => add_169
#   y_5 => add_177
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_497), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_499), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_501), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_503), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, %relu_50), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_171, [None, None, %convert_element_type_141, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_171, [None, None, %convert_element_type_141, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_171, [None, None, %clamp_max_12, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_171, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %clamp_max_14), kwargs = {})
#   %add_174 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_209), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %clamp_max_14), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_210), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_175, %add_174), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %clamp_max_15), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %mul_211), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_169, %add_176), kwargs = {})
#   %relu_60 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x5 = xindex
    x3 = ((xindex // 64) % 8)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x5), xmask)
    tmp20 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x5), xmask)
    tmp37 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr2 + (tmp8 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (tmp13 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp16
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp48 = tmp46 * tmp47
    tmp49 = tmp18 + tmp48
    tmp50 = tmp36 + tmp49
    tmp51 = tl.full([1], 0, tl.int32)
    tmp52 = triton_helpers.maximum(tmp51, tmp50)
    tl.store(in_out_ptr0 + (x5), tmp52, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2gyzh6cbqten2iviwejweg7gtkf5r6vwyjjr7xcjuhtbgtyqovb.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_25 => add_179, mul_213, mul_214, sub_84
#   input_26 => relu_61
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_513), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_515), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_213, %unsqueeze_517), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_214, %unsqueeze_519), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_179,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
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


# kernel path: inductor_cache/no/cno67zsxivipsphergwmjzxeb5cwwxzpy344hd6juhiyl2tstc7z.py
# Topologically Sorted Source Nodes: [input_28, input_30, y_6, y_7, residual_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_28 => add_181, mul_216, mul_217, sub_85
#   input_30 => add_183, mul_219, mul_220, sub_86
#   residual_4 => relu_62
#   y_6 => add_184
#   y_7 => add_185
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_521), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_523), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_216, %unsqueeze_525), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_217, %unsqueeze_527), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_531), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_219, %unsqueeze_533), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_220, %unsqueeze_535), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_181, %add_183), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %relu_58), kwargs = {})
#   %relu_62 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_185,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (x3), xmask)
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
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rj/crjwx6uugrcgug3oaywbets7kmfsk5e6vjuhnmqmvdaux2gkcunc.py
# Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_77 => add_472, mul_546, mul_547, sub_225
#   input_78 => relu_147
# Graph fragment:
#   %sub_225 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_1281), kwargs = {})
#   %mul_546 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_225, %unsqueeze_1283), kwargs = {})
#   %mul_547 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_546, %unsqueeze_1285), kwargs = {})
#   %add_472 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_547, %unsqueeze_1287), kwargs = {})
#   %relu_147 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_472,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
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


# kernel path: inductor_cache/xh/cxhfjpmunnfoafe7fprck7c27ebxah5fgthidhctdfcofhksyhmg.py
# Topologically Sorted Source Nodes: [out_520, out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_520 => add_536, mul_624, mul_625, sub_251
#   out_521 => add_537
#   out_522 => relu_173
# Graph fragment:
#   %sub_251 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_186, %unsqueeze_1489), kwargs = {})
#   %mul_624 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_251, %unsqueeze_1491), kwargs = {})
#   %mul_625 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_624, %unsqueeze_1493), kwargs = {})
#   %add_536 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_625, %unsqueeze_1495), kwargs = {})
#   %add_537 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_536, %relu_147), kwargs = {})
#   %relu_173 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_537,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfelujkwv3syohquxcoed3lrkube2evk2diyhqvdn7cklocma3wi.py
# Topologically Sorted Source Nodes: [input_84], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_84 => add_570, mul_661, mul_662, sub_270
# Graph fragment:
#   %sub_270 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_195, %unsqueeze_1561), kwargs = {})
#   %mul_661 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_270, %unsqueeze_1563), kwargs = {})
#   %mul_662 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_661, %unsqueeze_1565), kwargs = {})
#   %add_570 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_662, %unsqueeze_1567), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xu/cxu2osx2qmfjsft2ximtsl2mcstfmo6t5rvsytz5k3bgzk6cjuex.py
# Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_15 => convert_element_type_453
# Graph fragment:
#   %convert_element_type_453 : [num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_30, torch.int64), kwargs = {})
triton_poi_fused__to_copy_30 = async_compile.triton('triton_poi_fused__to_copy_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_30(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.06666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c2/cc2cfu3cpsl53zfllajart4tfv4zgmaycsy4fvyqtb4sbvp4skz5.py
# Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_15 => add_571, clamp_max_60
# Graph fragment:
#   %add_571 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_453, 1), kwargs = {})
#   %clamp_max_60 : [num_users=9] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_571, 1), kwargs = {})
triton_poi_fused_add_clamp_31 = async_compile.triton('triton_poi_fused_add_clamp_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_31(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.06666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yb/cybgdfxjrtyyxgxozelhsjs3y6oc7s3arqmcj6xljbn6kjotbo4g.py
# Topologically Sorted Source Nodes: [interpolate, interpolate_15], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate => convert_element_type_68, iota
#   interpolate_15 => clamp_max_62, clamp_min_60, clamp_min_62, mul_663, sub_271
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_68 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_663 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, 0.06666666666666667), kwargs = {})
#   %clamp_min_60 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_663, 0.0), kwargs = {})
#   %sub_271 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_60, %convert_element_type_455), kwargs = {})
#   %clamp_min_62 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_271, 0.0), kwargs = {})
#   %clamp_max_62 : [num_users=9] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_62, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_32 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_32(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.06666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xm/cxm5jb4e56cpopujzi5wiutf4ck7jmjponwttshn6qdswadksc5l.py
# Topologically Sorted Source Nodes: [out_457, out_458, out_459, interpolate_13, y_26, interpolate_14, y_27, interpolate_15, y_28, residual_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   interpolate_13 => _unsafe_index_52, _unsafe_index_53, _unsafe_index_54, _unsafe_index_55, add_557, add_558, add_559, mul_649, mul_650, mul_651, sub_260, sub_261, sub_263
#   interpolate_14 => _unsafe_index_56, _unsafe_index_57, _unsafe_index_58, _unsafe_index_59, add_565, add_566, add_567, mul_657, mul_658, mul_659, sub_266, sub_267, sub_269
#   interpolate_15 => _unsafe_index_60, _unsafe_index_61, _unsafe_index_62, _unsafe_index_63, add_573, add_574, add_575, mul_665, mul_666, mul_667, sub_272, sub_273, sub_275
#   out_457 => add_491, mul_570, mul_571, sub_233
#   out_458 => add_492
#   out_459 => relu_155
#   residual_14 => relu_180
#   y_26 => add_560
#   y_27 => add_568
#   y_28 => add_576
# Graph fragment:
#   %sub_233 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_168, %unsqueeze_1345), kwargs = {})
#   %mul_570 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %unsqueeze_1347), kwargs = {})
#   %mul_571 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_570, %unsqueeze_1349), kwargs = {})
#   %add_491 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_571, %unsqueeze_1351), kwargs = {})
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_491, %relu_153), kwargs = {})
#   %relu_155 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_492,), kwargs = {})
#   %_unsafe_index_52 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_554, [None, None, %convert_element_type_69, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_53 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_554, [None, None, %convert_element_type_69, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_54 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_554, [None, None, %clamp_max, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_55 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_554, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_260 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_53, %_unsafe_index_52), kwargs = {})
#   %mul_649 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_260, %clamp_max_2), kwargs = {})
#   %add_557 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_52, %mul_649), kwargs = {})
#   %sub_261 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_55, %_unsafe_index_54), kwargs = {})
#   %mul_650 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_261, %clamp_max_2), kwargs = {})
#   %add_558 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_54, %mul_650), kwargs = {})
#   %sub_263 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_558, %add_557), kwargs = {})
#   %mul_651 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_263, %clamp_max_3), kwargs = {})
#   %add_559 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_557, %mul_651), kwargs = {})
#   %add_560 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_155, %add_559), kwargs = {})
#   %_unsafe_index_56 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_562, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_57 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_562, [None, None, %convert_element_type_133, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_58 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_562, [None, None, %clamp_max_8, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_59 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_562, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_266 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_57, %_unsafe_index_56), kwargs = {})
#   %mul_657 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_266, %clamp_max_10), kwargs = {})
#   %add_565 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_56, %mul_657), kwargs = {})
#   %sub_267 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_59, %_unsafe_index_58), kwargs = {})
#   %mul_658 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_267, %clamp_max_10), kwargs = {})
#   %add_566 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_58, %mul_658), kwargs = {})
#   %sub_269 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_566, %add_565), kwargs = {})
#   %mul_659 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_269, %clamp_max_11), kwargs = {})
#   %add_567 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_565, %mul_659), kwargs = {})
#   %add_568 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_560, %add_567), kwargs = {})
#   %_unsafe_index_60 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_570, [None, None, %convert_element_type_453, %convert_element_type_455]), kwargs = {})
#   %_unsafe_index_61 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_570, [None, None, %convert_element_type_453, %clamp_max_61]), kwargs = {})
#   %_unsafe_index_62 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_570, [None, None, %clamp_max_60, %convert_element_type_455]), kwargs = {})
#   %_unsafe_index_63 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_570, [None, None, %clamp_max_60, %clamp_max_61]), kwargs = {})
#   %sub_272 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_61, %_unsafe_index_60), kwargs = {})
#   %mul_665 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_272, %clamp_max_62), kwargs = {})
#   %add_573 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_60, %mul_665), kwargs = {})
#   %sub_273 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_63, %_unsafe_index_62), kwargs = {})
#   %mul_666 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_273, %clamp_max_62), kwargs = {})
#   %add_574 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_62, %mul_666), kwargs = {})
#   %sub_275 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_574, %add_573), kwargs = {})
#   %mul_667 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_275, %clamp_max_63), kwargs = {})
#   %add_575 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_573, %mul_667), kwargs = {})
#   %add_576 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_568, %add_575), kwargs = {})
#   %relu_180 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_576,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*i64', 'in_ptr21': '*i64', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*fp32', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x6 = xindex
    x1 = ((xindex // 256) % 4)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x3), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x3), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr17 + (x3), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr20 + (x4), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr21 + (x3), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr23 + (x3), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr24 + (x3), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr25 + (x4), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr26 + (x4), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp40 = tmp39 + tmp21
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr8 + (tmp28 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (tmp33 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45 * tmp36
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47 - tmp38
    tmp50 = tmp48 * tmp49
    tmp51 = tmp38 + tmp50
    tmp52 = tmp19 + tmp51
    tmp54 = tl.full([XBLOCK], 4, tl.int32)
    tmp55 = tmp53 + tmp54
    tmp56 = tmp53 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp53)
    tmp59 = tmp58 + tmp54
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tmp62 = tl.load(in_ptr15 + (tmp61 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp64 = tmp63 + tmp54
    tmp65 = tmp63 < 0
    tmp66 = tl.where(tmp65, tmp64, tmp63)
    tmp67 = tl.load(in_ptr15 + (tmp66 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp68 = tmp67 - tmp62
    tmp70 = tmp68 * tmp69
    tmp71 = tmp62 + tmp70
    tmp73 = tmp72 + tmp54
    tmp74 = tmp72 < 0
    tmp75 = tl.where(tmp74, tmp73, tmp72)
    tmp76 = tl.load(in_ptr15 + (tmp61 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr15 + (tmp66 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp78 = tmp77 - tmp76
    tmp79 = tmp78 * tmp69
    tmp80 = tmp76 + tmp79
    tmp81 = tmp80 - tmp71
    tmp83 = tmp81 * tmp82
    tmp84 = tmp71 + tmp83
    tmp85 = tmp52 + tmp84
    tmp87 = tl.full([XBLOCK], 2, tl.int32)
    tmp88 = tmp86 + tmp87
    tmp89 = tmp86 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp86)
    tmp92 = tmp91 + tmp87
    tmp93 = tmp91 < 0
    tmp94 = tl.where(tmp93, tmp92, tmp91)
    tmp95 = tl.load(in_ptr22 + (tmp94 + 2*tmp90 + 4*x5), None, eviction_policy='evict_last')
    tmp97 = tmp96 + tmp87
    tmp98 = tmp96 < 0
    tmp99 = tl.where(tmp98, tmp97, tmp96)
    tmp100 = tl.load(in_ptr22 + (tmp99 + 2*tmp90 + 4*x5), None, eviction_policy='evict_last')
    tmp101 = tmp100 - tmp95
    tmp103 = tmp101 * tmp102
    tmp104 = tmp95 + tmp103
    tmp106 = tmp105 + tmp87
    tmp107 = tmp105 < 0
    tmp108 = tl.where(tmp107, tmp106, tmp105)
    tmp109 = tl.load(in_ptr22 + (tmp94 + 2*tmp108 + 4*x5), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr22 + (tmp99 + 2*tmp108 + 4*x5), None, eviction_policy='evict_last')
    tmp111 = tmp110 - tmp109
    tmp112 = tmp111 * tmp102
    tmp113 = tmp109 + tmp112
    tmp114 = tmp113 - tmp104
    tmp116 = tmp114 * tmp115
    tmp117 = tmp104 + tmp116
    tmp118 = tmp85 + tmp117
    tmp119 = triton_helpers.maximum(tmp18, tmp118)
    tl.store(out_ptr0 + (x6), tmp19, None)
    tl.store(in_out_ptr0 + (x6), tmp119, None)
''', device_str='cuda')


# kernel path: inductor_cache/qo/cqos6oa26unywxigucenypzavprh52boej3cx4l5zsyblhgebjl5.py
# Topologically Sorted Source Nodes: [input_90], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_90 => add_589, mul_680, mul_681, sub_283
# Graph fragment:
#   %sub_283 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_1585), kwargs = {})
#   %mul_680 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_283, %unsqueeze_1587), kwargs = {})
#   %mul_681 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_680, %unsqueeze_1589), kwargs = {})
#   %add_589 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_681, %unsqueeze_1591), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 8)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7y/c7yoo46khuqqjgupoprvnblsihr2meu55xtnx4zec7y77dpi25pt.py
# Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_17 => convert_element_type_467
# Graph fragment:
#   %convert_element_type_467 : [num_users=9] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_34, torch.int64), kwargs = {})
triton_poi_fused__to_copy_35 = async_compile.triton('triton_poi_fused__to_copy_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_35(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.14285714285714285
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxbaxf2mzn4wbgdftzjuttbbnc452ynzuysp3oqpksparzlc4jj.py
# Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_17 => add_590, clamp_max_68
# Graph fragment:
#   %add_590 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_467, 1), kwargs = {})
#   %clamp_max_68 : [num_users=7] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_590, 1), kwargs = {})
triton_poi_fused_add_clamp_36 = async_compile.triton('triton_poi_fused_add_clamp_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_36(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.14285714285714285
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2p/c2pd3d5l3v4b43dwva7cx5ncy6e6ofm46noausyitmdaadyqlt5i.py
# Topologically Sorted Source Nodes: [interpolate_3, interpolate_17], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate_17 => clamp_max_70, clamp_min_68, clamp_min_70, mul_682, sub_284
#   interpolate_3 => convert_element_type_140, iota_6
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_140 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %mul_682 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_140, 0.14285714285714285), kwargs = {})
#   %clamp_min_68 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_682, 0.0), kwargs = {})
#   %sub_284 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_68, %convert_element_type_469), kwargs = {})
#   %clamp_min_70 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_284, 0.0), kwargs = {})
#   %clamp_max_70 : [num_users=7] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_70, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_37 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_37(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.14285714285714285
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7hkx53wsyscmzw7utw4adnwl23alled4cumfyrmlvnsfmg7amji.py
# Topologically Sorted Source Nodes: [input_86, y_29, interpolate_16, y_30, interpolate_17, y_31, residual_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
# Source node to ATen node mapping:
#   input_86 => add_578, mul_669, mul_670, sub_276
#   interpolate_16 => _unsafe_index_64, _unsafe_index_65, _unsafe_index_66, _unsafe_index_67, add_584, add_585, add_586, mul_676, mul_677, mul_678, sub_279, sub_280, sub_282
#   interpolate_17 => _unsafe_index_68, _unsafe_index_69, _unsafe_index_70, _unsafe_index_71, add_592, add_593, add_594, mul_684, mul_685, mul_686, sub_285, sub_286, sub_288
#   residual_15 => relu_181
#   y_29 => add_579
#   y_30 => add_587
#   y_31 => add_595
# Graph fragment:
#   %sub_276 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_196, %unsqueeze_1569), kwargs = {})
#   %mul_669 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_276, %unsqueeze_1571), kwargs = {})
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_669, %unsqueeze_1573), kwargs = {})
#   %add_578 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_670, %unsqueeze_1575), kwargs = {})
#   %add_579 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_578, %relu_163), kwargs = {})
#   %_unsafe_index_64 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_581, [None, None, %convert_element_type_141, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_65 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_581, [None, None, %convert_element_type_141, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_66 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_581, [None, None, %clamp_max_12, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_67 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_581, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_279 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_65, %_unsafe_index_64), kwargs = {})
#   %mul_676 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_279, %clamp_max_14), kwargs = {})
#   %add_584 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_64, %mul_676), kwargs = {})
#   %sub_280 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_67, %_unsafe_index_66), kwargs = {})
#   %mul_677 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_280, %clamp_max_14), kwargs = {})
#   %add_585 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_66, %mul_677), kwargs = {})
#   %sub_282 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_585, %add_584), kwargs = {})
#   %mul_678 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_282, %clamp_max_15), kwargs = {})
#   %add_586 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_584, %mul_678), kwargs = {})
#   %add_587 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_579, %add_586), kwargs = {})
#   %_unsafe_index_68 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_589, [None, None, %convert_element_type_467, %convert_element_type_469]), kwargs = {})
#   %_unsafe_index_69 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_589, [None, None, %convert_element_type_467, %clamp_max_69]), kwargs = {})
#   %_unsafe_index_70 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_589, [None, None, %clamp_max_68, %convert_element_type_469]), kwargs = {})
#   %_unsafe_index_71 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_589, [None, None, %clamp_max_68, %clamp_max_69]), kwargs = {})
#   %sub_285 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_69, %_unsafe_index_68), kwargs = {})
#   %mul_684 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_285, %clamp_max_70), kwargs = {})
#   %add_592 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_68, %mul_684), kwargs = {})
#   %sub_286 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_71, %_unsafe_index_70), kwargs = {})
#   %mul_685 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_286, %clamp_max_70), kwargs = {})
#   %add_593 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_70, %mul_685), kwargs = {})
#   %sub_288 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_593, %add_592), kwargs = {})
#   %mul_686 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_288, %clamp_max_71), kwargs = {})
#   %add_594 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_592, %mul_686), kwargs = {})
#   %add_595 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_587, %add_594), kwargs = {})
#   %relu_181 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_595,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x5 = xindex
    x3 = ((xindex // 64) % 8)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x5), xmask)
    tmp20 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x5), xmask)
    tmp37 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr14 + (x0), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr16 + (x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr17 + (x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr2 + (tmp8 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (tmp13 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp16
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp48 = tmp46 * tmp47
    tmp49 = tmp18 + tmp48
    tmp50 = tmp36 + tmp49
    tmp52 = tl.full([XBLOCK], 2, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tmp57 = tmp56 + tmp52
    tmp58 = tmp56 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp56)
    tmp60 = tl.load(in_ptr15 + (tmp59 + 2*tmp55 + 4*x2), xmask, eviction_policy='evict_last')
    tmp62 = tmp61 + tmp52
    tmp63 = tmp61 < 0
    tmp64 = tl.where(tmp63, tmp62, tmp61)
    tmp65 = tl.load(in_ptr15 + (tmp64 + 2*tmp55 + 4*x2), xmask, eviction_policy='evict_last')
    tmp66 = tmp65 - tmp60
    tmp68 = tmp66 * tmp67
    tmp69 = tmp60 + tmp68
    tmp71 = tmp70 + tmp52
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tmp74 = tl.load(in_ptr15 + (tmp59 + 2*tmp73 + 4*x2), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr15 + (tmp64 + 2*tmp73 + 4*x2), xmask, eviction_policy='evict_last')
    tmp76 = tmp75 - tmp74
    tmp77 = tmp76 * tmp67
    tmp78 = tmp74 + tmp77
    tmp79 = tmp78 - tmp69
    tmp81 = tmp79 * tmp80
    tmp82 = tmp69 + tmp81
    tmp83 = tmp50 + tmp82
    tmp84 = tl.full([1], 0, tl.int32)
    tmp85 = triton_helpers.maximum(tmp84, tmp83)
    tl.store(in_out_ptr0 + (x5), tmp85, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hx/chx67kadz6p2vmplubfmpc36bx4uy3hligfjn4fpwmdk3x7c326x.py
# Topologically Sorted Source Nodes: [input_99], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_99 => add_605, mul_697, mul_698, sub_292
# Graph fragment:
#   %sub_292 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_1617), kwargs = {})
#   %mul_697 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_292, %unsqueeze_1619), kwargs = {})
#   %mul_698 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_697, %unsqueeze_1621), kwargs = {})
#   %add_605 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_698, %unsqueeze_1623), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oz/cozs3hy7tkwn5ullk62wlztrsrlc7mkwaolp2qny7da52f5tfxiq.py
# Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_18 => convert_element_type_479
# Graph fragment:
#   %convert_element_type_479 : [num_users=9] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_36, torch.int64), kwargs = {})
triton_poi_fused__to_copy_40 = async_compile.triton('triton_poi_fused__to_copy_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_40(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgjqyzcruz7z6gq3affbgcdyqzdaiucwc5gykwtuah3xnlgjjkp4.py
# Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_18 => add_606, clamp_max_72
# Graph fragment:
#   %add_606 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_479, 1), kwargs = {})
#   %clamp_max_72 : [num_users=7] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_606, 1), kwargs = {})
triton_poi_fused_add_clamp_41 = async_compile.triton('triton_poi_fused_add_clamp_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujb7pkcbhglbcpxq6eiz3ruz6nepy5v6azqp5el667gb4k7qj5b.py
# Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   interpolate_18 => clamp_max_74, clamp_min_72, clamp_min_74, convert_element_type_478, iota_36, mul_699, sub_293
# Graph fragment:
#   %iota_36 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_478 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_36, torch.float32), kwargs = {})
#   %mul_699 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_478, 0.3333333333333333), kwargs = {})
#   %clamp_min_72 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_699, 0.0), kwargs = {})
#   %sub_293 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_72, %convert_element_type_481), kwargs = {})
#   %clamp_min_74 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_293, 0.0), kwargs = {})
#   %clamp_max_74 : [num_users=7] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_74, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_42 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nh/cnhornliljh3l6g4f32kda7p4gognefy5xxbcbz6dnhl3zpzesl7.py
# Topologically Sorted Source Nodes: [input_95, input_97, y_32, y_33, interpolate_18, y_34, residual_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
# Source node to ATen node mapping:
#   input_95 => add_599, mul_691, mul_692, sub_290
#   input_97 => add_601, mul_694, mul_695, sub_291
#   interpolate_18 => _unsafe_index_72, _unsafe_index_73, _unsafe_index_74, _unsafe_index_75, add_608, add_609, add_610, mul_701, mul_702, mul_703, sub_294, sub_295, sub_297
#   residual_16 => relu_183
#   y_32 => add_602
#   y_33 => add_603
#   y_34 => add_611
# Graph fragment:
#   %sub_290 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_200, %unsqueeze_1601), kwargs = {})
#   %mul_691 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_290, %unsqueeze_1603), kwargs = {})
#   %mul_692 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_691, %unsqueeze_1605), kwargs = {})
#   %add_599 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_692, %unsqueeze_1607), kwargs = {})
#   %sub_291 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1609), kwargs = {})
#   %mul_694 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_291, %unsqueeze_1611), kwargs = {})
#   %mul_695 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_694, %unsqueeze_1613), kwargs = {})
#   %add_601 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_695, %unsqueeze_1615), kwargs = {})
#   %add_602 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_599, %add_601), kwargs = {})
#   %add_603 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_602, %relu_171), kwargs = {})
#   %_unsafe_index_72 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_605, [None, None, %convert_element_type_479, %convert_element_type_481]), kwargs = {})
#   %_unsafe_index_73 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_605, [None, None, %convert_element_type_479, %clamp_max_73]), kwargs = {})
#   %_unsafe_index_74 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_605, [None, None, %clamp_max_72, %convert_element_type_481]), kwargs = {})
#   %_unsafe_index_75 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_605, [None, None, %clamp_max_72, %clamp_max_73]), kwargs = {})
#   %sub_294 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_73, %_unsafe_index_72), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_294, %clamp_max_74), kwargs = {})
#   %add_608 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_72, %mul_701), kwargs = {})
#   %sub_295 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_75, %_unsafe_index_74), kwargs = {})
#   %mul_702 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_295, %clamp_max_74), kwargs = {})
#   %add_609 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_74, %mul_702), kwargs = {})
#   %sub_297 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_609, %add_608), kwargs = {})
#   %mul_703 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_297, %clamp_max_75), kwargs = {})
#   %add_610 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_608, %mul_703), kwargs = {})
#   %add_611 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_603, %add_610), kwargs = {})
#   %relu_183 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_611,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 16) % 16)
    x4 = ((xindex // 4) % 4)
    x3 = (xindex % 4)
    x5 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x3), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + (x3), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr15 + (x6), xmask)
    tmp51 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
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
    tmp31 = tl.full([XBLOCK], 2, tl.int32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tmp36 = tmp35 + tmp31
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tmp39 = tl.load(in_ptr12 + (tmp38 + 2*tmp34 + 4*x5), xmask, eviction_policy='evict_last')
    tmp41 = tmp40 + tmp31
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tl.load(in_ptr12 + (tmp43 + 2*tmp34 + 4*x5), xmask, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp39
    tmp47 = tmp45 * tmp46
    tmp48 = tmp39 + tmp47
    tmp50 = tmp29 + tmp49
    tmp52 = tmp51 + tmp31
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tmp55 = tl.load(in_ptr12 + (tmp38 + 2*tmp54 + 4*x5), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (tmp43 + 2*tmp54 + 4*x5), xmask, eviction_policy='evict_last')
    tmp57 = tmp56 - tmp55
    tmp58 = tmp57 * tmp46
    tmp59 = tmp55 + tmp58
    tmp60 = tmp59 - tmp48
    tmp62 = tmp60 * tmp61
    tmp63 = tmp48 + tmp62
    tmp64 = tmp50 + tmp63
    tmp65 = tl.full([1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tl.store(in_out_ptr0 + (x6), tmp66, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gb/cgbpaakhqpalynhggfbkj7dkhiinglbqtdvgbdkpoljx663jeb4t.py
# Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_104 => add_615, mul_708, mul_709, sub_299
#   input_105 => relu_185
# Graph fragment:
#   %sub_299 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1633), kwargs = {})
#   %mul_708 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_299, %unsqueeze_1635), kwargs = {})
#   %mul_709 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_708, %unsqueeze_1637), kwargs = {})
#   %add_615 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_709, %unsqueeze_1639), kwargs = {})
#   %relu_185 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_615,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
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


# kernel path: inductor_cache/b5/cb5mfsdyt7y7kokmk53cpgfpiei5ppmog7x6hn3ctnwj2ana5sbb.py
# Topologically Sorted Source Nodes: [input_109, input_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_109 => add_619, mul_714, mul_715, sub_301
#   input_110 => relu_186
# Graph fragment:
#   %sub_301 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_206, %unsqueeze_1649), kwargs = {})
#   %mul_714 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_301, %unsqueeze_1651), kwargs = {})
#   %mul_715 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_714, %unsqueeze_1653), kwargs = {})
#   %add_619 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_715, %unsqueeze_1655), kwargs = {})
#   %relu_186 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_619,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 8)
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


# kernel path: inductor_cache/a3/ca3pnxryflb5m3df5clinp5voszw5icsohavlbz7v6ms67cln2oi.py
# Topologically Sorted Source Nodes: [input_107, input_112, y_35, input_114, y_36, y_37, residual_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_107 => add_617, mul_711, mul_712, sub_300
#   input_112 => add_621, mul_717, mul_718, sub_302
#   input_114 => add_624, mul_720, mul_721, sub_303
#   residual_17 => relu_187
#   y_35 => add_622
#   y_36 => add_625
#   y_37 => add_626
# Graph fragment:
#   %sub_300 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_205, %unsqueeze_1641), kwargs = {})
#   %mul_711 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_300, %unsqueeze_1643), kwargs = {})
#   %mul_712 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_711, %unsqueeze_1645), kwargs = {})
#   %add_617 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_712, %unsqueeze_1647), kwargs = {})
#   %sub_302 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_207, %unsqueeze_1657), kwargs = {})
#   %mul_717 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_302, %unsqueeze_1659), kwargs = {})
#   %mul_718 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_717, %unsqueeze_1661), kwargs = {})
#   %add_621 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_718, %unsqueeze_1663), kwargs = {})
#   %add_622 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_617, %add_621), kwargs = {})
#   %sub_303 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_208, %unsqueeze_1665), kwargs = {})
#   %mul_720 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_303, %unsqueeze_1667), kwargs = {})
#   %mul_721 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_720, %unsqueeze_1669), kwargs = {})
#   %add_624 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_721, %unsqueeze_1671), kwargs = {})
#   %add_625 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_622, %add_624), kwargs = {})
#   %add_626 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_625, %relu_179), kwargs = {})
#   %relu_187 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_626,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (x3), xmask)
    tmp31 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr15 + (x3), xmask)
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(in_out_ptr0 + (x3), tmp47, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtfthvmpexhbm454gjf2vg3byy2xdc6haorhn7lic6dzzp6kkbf.py
# Topologically Sorted Source Nodes: [out_681, out_682, out_683, interpolate_25, y_50, interpolate_26, y_51, interpolate_27, y_52, relu_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.threshold_backward]
# Source node to ATen node mapping:
#   interpolate_25 => _unsafe_index_100, _unsafe_index_101, _unsafe_index_102, _unsafe_index_103, add_865, add_866, add_867, mul_997, mul_998, mul_999, sub_416, sub_417, sub_419
#   interpolate_26 => _unsafe_index_104, _unsafe_index_105, _unsafe_index_106, _unsafe_index_107, add_873, add_874, add_875, mul_1005, mul_1006, mul_1007, sub_422, sub_423, sub_425
#   interpolate_27 => _unsafe_index_108, _unsafe_index_109, _unsafe_index_110, _unsafe_index_111, add_881, add_882, add_883, mul_1013, mul_1014, mul_1015, sub_428, sub_429, sub_431
#   out_681 => add_799, mul_918, mul_919, sub_389
#   out_682 => add_800
#   out_683 => relu_235
#   relu_260 => relu_260
#   y_50 => add_868
#   y_51 => add_876
#   y_52 => add_884
# Graph fragment:
#   %sub_389 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_264, %unsqueeze_2113), kwargs = {})
#   %mul_918 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_389, %unsqueeze_2115), kwargs = {})
#   %mul_919 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_918, %unsqueeze_2117), kwargs = {})
#   %add_799 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_919, %unsqueeze_2119), kwargs = {})
#   %add_800 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_799, %relu_233), kwargs = {})
#   %relu_235 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_800,), kwargs = {})
#   %_unsafe_index_100 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_862, [None, None, %convert_element_type_69, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_101 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_862, [None, None, %convert_element_type_69, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_102 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_862, [None, None, %clamp_max, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_103 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_862, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_416 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_101, %_unsafe_index_100), kwargs = {})
#   %mul_997 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_416, %clamp_max_2), kwargs = {})
#   %add_865 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_100, %mul_997), kwargs = {})
#   %sub_417 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_103, %_unsafe_index_102), kwargs = {})
#   %mul_998 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_417, %clamp_max_2), kwargs = {})
#   %add_866 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_102, %mul_998), kwargs = {})
#   %sub_419 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_866, %add_865), kwargs = {})
#   %mul_999 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_419, %clamp_max_3), kwargs = {})
#   %add_867 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_865, %mul_999), kwargs = {})
#   %add_868 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_235, %add_867), kwargs = {})
#   %_unsafe_index_104 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_870, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_105 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_870, [None, None, %convert_element_type_133, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_106 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_870, [None, None, %clamp_max_8, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_107 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_870, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_422 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_105, %_unsafe_index_104), kwargs = {})
#   %mul_1005 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_422, %clamp_max_10), kwargs = {})
#   %add_873 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_104, %mul_1005), kwargs = {})
#   %sub_423 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_107, %_unsafe_index_106), kwargs = {})
#   %mul_1006 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_423, %clamp_max_10), kwargs = {})
#   %add_874 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_106, %mul_1006), kwargs = {})
#   %sub_425 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_874, %add_873), kwargs = {})
#   %mul_1007 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_425, %clamp_max_11), kwargs = {})
#   %add_875 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_873, %mul_1007), kwargs = {})
#   %add_876 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_868, %add_875), kwargs = {})
#   %_unsafe_index_108 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_878, [None, None, %convert_element_type_453, %convert_element_type_455]), kwargs = {})
#   %_unsafe_index_109 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_878, [None, None, %convert_element_type_453, %clamp_max_61]), kwargs = {})
#   %_unsafe_index_110 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_878, [None, None, %clamp_max_60, %convert_element_type_455]), kwargs = {})
#   %_unsafe_index_111 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_878, [None, None, %clamp_max_60, %clamp_max_61]), kwargs = {})
#   %sub_428 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_109, %_unsafe_index_108), kwargs = {})
#   %mul_1013 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_428, %clamp_max_62), kwargs = {})
#   %add_881 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_108, %mul_1013), kwargs = {})
#   %sub_429 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_111, %_unsafe_index_110), kwargs = {})
#   %mul_1014 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_429, %clamp_max_62), kwargs = {})
#   %add_882 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_110, %mul_1014), kwargs = {})
#   %sub_431 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_882, %add_881), kwargs = {})
#   %mul_1015 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_431, %clamp_max_63), kwargs = {})
#   %add_883 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_881, %mul_1015), kwargs = {})
#   %add_884 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_876, %add_883), kwargs = {})
#   %relu_260 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_884,), kwargs = {})
#   %le_16 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_260, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*i64', 'in_ptr21': '*i64', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*fp32', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr0, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x6 = xindex
    x1 = ((xindex // 256) % 4)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x3), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x3), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr17 + (x3), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr20 + (x4), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr21 + (x3), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr23 + (x3), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr24 + (x3), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr25 + (x4), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr26 + (x4), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp40 = tmp39 + tmp21
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr8 + (tmp28 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (tmp33 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45 * tmp36
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47 - tmp38
    tmp50 = tmp48 * tmp49
    tmp51 = tmp38 + tmp50
    tmp52 = tmp19 + tmp51
    tmp54 = tl.full([XBLOCK], 4, tl.int32)
    tmp55 = tmp53 + tmp54
    tmp56 = tmp53 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp53)
    tmp59 = tmp58 + tmp54
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tmp62 = tl.load(in_ptr15 + (tmp61 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp64 = tmp63 + tmp54
    tmp65 = tmp63 < 0
    tmp66 = tl.where(tmp65, tmp64, tmp63)
    tmp67 = tl.load(in_ptr15 + (tmp66 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp68 = tmp67 - tmp62
    tmp70 = tmp68 * tmp69
    tmp71 = tmp62 + tmp70
    tmp73 = tmp72 + tmp54
    tmp74 = tmp72 < 0
    tmp75 = tl.where(tmp74, tmp73, tmp72)
    tmp76 = tl.load(in_ptr15 + (tmp61 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr15 + (tmp66 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp78 = tmp77 - tmp76
    tmp79 = tmp78 * tmp69
    tmp80 = tmp76 + tmp79
    tmp81 = tmp80 - tmp71
    tmp83 = tmp81 * tmp82
    tmp84 = tmp71 + tmp83
    tmp85 = tmp52 + tmp84
    tmp87 = tl.full([XBLOCK], 2, tl.int32)
    tmp88 = tmp86 + tmp87
    tmp89 = tmp86 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp86)
    tmp92 = tmp91 + tmp87
    tmp93 = tmp91 < 0
    tmp94 = tl.where(tmp93, tmp92, tmp91)
    tmp95 = tl.load(in_ptr22 + (tmp94 + 2*tmp90 + 4*x5), None, eviction_policy='evict_last')
    tmp97 = tmp96 + tmp87
    tmp98 = tmp96 < 0
    tmp99 = tl.where(tmp98, tmp97, tmp96)
    tmp100 = tl.load(in_ptr22 + (tmp99 + 2*tmp90 + 4*x5), None, eviction_policy='evict_last')
    tmp101 = tmp100 - tmp95
    tmp103 = tmp101 * tmp102
    tmp104 = tmp95 + tmp103
    tmp106 = tmp105 + tmp87
    tmp107 = tmp105 < 0
    tmp108 = tl.where(tmp107, tmp106, tmp105)
    tmp109 = tl.load(in_ptr22 + (tmp94 + 2*tmp108 + 4*x5), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr22 + (tmp99 + 2*tmp108 + 4*x5), None, eviction_policy='evict_last')
    tmp111 = tmp110 - tmp109
    tmp112 = tmp111 * tmp102
    tmp113 = tmp109 + tmp112
    tmp114 = tmp113 - tmp104
    tmp116 = tmp114 * tmp115
    tmp117 = tmp104 + tmp116
    tmp118 = tmp85 + tmp117
    tmp119 = triton_helpers.maximum(tmp18, tmp118)
    tmp120 = 0.0
    tmp121 = tmp119 <= tmp120
    tl.store(out_ptr0 + (x6), tmp19, None)
    tl.store(in_out_ptr0 + (x6), tmp118, None)
    tl.store(out_ptr3 + (x6), tmp121, None)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cji4ij7plv4weghd5jkqjhs67buoqazujfh4ay4kd433gj2hwtll.py
# Topologically Sorted Source Nodes: [input_158, y_53, interpolate_28, y_54, interpolate_29, y_55, relu_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_158 => add_886, mul_1017, mul_1018, sub_432
#   interpolate_28 => _unsafe_index_112, _unsafe_index_113, _unsafe_index_114, _unsafe_index_115, add_892, add_893, add_894, mul_1024, mul_1025, mul_1026, sub_435, sub_436, sub_438
#   interpolate_29 => _unsafe_index_116, _unsafe_index_117, _unsafe_index_118, _unsafe_index_119, add_900, add_901, add_902, mul_1032, mul_1033, mul_1034, sub_441, sub_442, sub_444
#   relu_261 => relu_261
#   y_53 => add_887
#   y_54 => add_895
#   y_55 => add_903
# Graph fragment:
#   %sub_432 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_292, %unsqueeze_2337), kwargs = {})
#   %mul_1017 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_432, %unsqueeze_2339), kwargs = {})
#   %mul_1018 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1017, %unsqueeze_2341), kwargs = {})
#   %add_886 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1018, %unsqueeze_2343), kwargs = {})
#   %add_887 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_886, %relu_243), kwargs = {})
#   %_unsafe_index_112 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_889, [None, None, %convert_element_type_141, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_113 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_889, [None, None, %convert_element_type_141, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_114 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_889, [None, None, %clamp_max_12, %convert_element_type_143]), kwargs = {})
#   %_unsafe_index_115 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_889, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_435 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_113, %_unsafe_index_112), kwargs = {})
#   %mul_1024 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_435, %clamp_max_14), kwargs = {})
#   %add_892 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_112, %mul_1024), kwargs = {})
#   %sub_436 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_115, %_unsafe_index_114), kwargs = {})
#   %mul_1025 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_436, %clamp_max_14), kwargs = {})
#   %add_893 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_114, %mul_1025), kwargs = {})
#   %sub_438 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_893, %add_892), kwargs = {})
#   %mul_1026 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_438, %clamp_max_15), kwargs = {})
#   %add_894 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_892, %mul_1026), kwargs = {})
#   %add_895 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_887, %add_894), kwargs = {})
#   %_unsafe_index_116 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_897, [None, None, %convert_element_type_467, %convert_element_type_469]), kwargs = {})
#   %_unsafe_index_117 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_897, [None, None, %convert_element_type_467, %clamp_max_69]), kwargs = {})
#   %_unsafe_index_118 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_897, [None, None, %clamp_max_68, %convert_element_type_469]), kwargs = {})
#   %_unsafe_index_119 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_897, [None, None, %clamp_max_68, %clamp_max_69]), kwargs = {})
#   %sub_441 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_117, %_unsafe_index_116), kwargs = {})
#   %mul_1032 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_441, %clamp_max_70), kwargs = {})
#   %add_900 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_116, %mul_1032), kwargs = {})
#   %sub_442 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_119, %_unsafe_index_118), kwargs = {})
#   %mul_1033 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_442, %clamp_max_70), kwargs = {})
#   %add_901 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_118, %mul_1033), kwargs = {})
#   %sub_444 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_901, %add_900), kwargs = {})
#   %mul_1034 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_444, %clamp_max_71), kwargs = {})
#   %add_902 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_900, %mul_1034), kwargs = {})
#   %add_903 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_895, %add_902), kwargs = {})
#   %relu_261 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_903,), kwargs = {})
#   %le_15 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_261, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x5 = xindex
    x3 = ((xindex // 64) % 8)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x5), xmask)
    tmp20 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x5), xmask)
    tmp37 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr14 + (x0), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr16 + (x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr17 + (x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr2 + (tmp8 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (tmp13 + 4*tmp40 + 16*x2), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp16
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp48 = tmp46 * tmp47
    tmp49 = tmp18 + tmp48
    tmp50 = tmp36 + tmp49
    tmp52 = tl.full([XBLOCK], 2, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tmp57 = tmp56 + tmp52
    tmp58 = tmp56 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp56)
    tmp60 = tl.load(in_ptr15 + (tmp59 + 2*tmp55 + 4*x2), xmask, eviction_policy='evict_last')
    tmp62 = tmp61 + tmp52
    tmp63 = tmp61 < 0
    tmp64 = tl.where(tmp63, tmp62, tmp61)
    tmp65 = tl.load(in_ptr15 + (tmp64 + 2*tmp55 + 4*x2), xmask, eviction_policy='evict_last')
    tmp66 = tmp65 - tmp60
    tmp68 = tmp66 * tmp67
    tmp69 = tmp60 + tmp68
    tmp71 = tmp70 + tmp52
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tmp74 = tl.load(in_ptr15 + (tmp59 + 2*tmp73 + 4*x2), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr15 + (tmp64 + 2*tmp73 + 4*x2), xmask, eviction_policy='evict_last')
    tmp76 = tmp75 - tmp74
    tmp77 = tmp76 * tmp67
    tmp78 = tmp74 + tmp77
    tmp79 = tmp78 - tmp69
    tmp81 = tmp79 * tmp80
    tmp82 = tmp69 + tmp81
    tmp83 = tmp50 + tmp82
    tmp84 = tl.full([1], 0, tl.int32)
    tmp85 = triton_helpers.maximum(tmp84, tmp83)
    tmp86 = 0.0
    tmp87 = tmp85 <= tmp86
    tl.store(in_out_ptr0 + (x5), tmp83, xmask)
    tl.store(out_ptr1 + (x5), tmp87, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/em/cemlumpvaej4htigiygi2ffdowzfqca5v7zrtkkfq6xoiiirntyj.py
# Topologically Sorted Source Nodes: [input_167, input_169, y_56, y_57, interpolate_30, y_58, relu_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_167 => add_907, mul_1039, mul_1040, sub_446
#   input_169 => add_909, mul_1042, mul_1043, sub_447
#   interpolate_30 => _unsafe_index_120, _unsafe_index_121, _unsafe_index_122, _unsafe_index_123, add_916, add_917, add_918, mul_1049, mul_1050, mul_1051, sub_450, sub_451, sub_453
#   relu_263 => relu_263
#   y_56 => add_910
#   y_57 => add_911
#   y_58 => add_919
# Graph fragment:
#   %sub_446 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_296, %unsqueeze_2369), kwargs = {})
#   %mul_1039 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_446, %unsqueeze_2371), kwargs = {})
#   %mul_1040 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1039, %unsqueeze_2373), kwargs = {})
#   %add_907 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1040, %unsqueeze_2375), kwargs = {})
#   %sub_447 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_297, %unsqueeze_2377), kwargs = {})
#   %mul_1042 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_447, %unsqueeze_2379), kwargs = {})
#   %mul_1043 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1042, %unsqueeze_2381), kwargs = {})
#   %add_909 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1043, %unsqueeze_2383), kwargs = {})
#   %add_910 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_907, %add_909), kwargs = {})
#   %add_911 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_910, %relu_251), kwargs = {})
#   %_unsafe_index_120 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_913, [None, None, %convert_element_type_479, %convert_element_type_481]), kwargs = {})
#   %_unsafe_index_121 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_913, [None, None, %convert_element_type_479, %clamp_max_73]), kwargs = {})
#   %_unsafe_index_122 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_913, [None, None, %clamp_max_72, %convert_element_type_481]), kwargs = {})
#   %_unsafe_index_123 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_913, [None, None, %clamp_max_72, %clamp_max_73]), kwargs = {})
#   %sub_450 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_121, %_unsafe_index_120), kwargs = {})
#   %mul_1049 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_450, %clamp_max_74), kwargs = {})
#   %add_916 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_120, %mul_1049), kwargs = {})
#   %sub_451 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_123, %_unsafe_index_122), kwargs = {})
#   %mul_1050 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_451, %clamp_max_74), kwargs = {})
#   %add_917 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_122, %mul_1050), kwargs = {})
#   %sub_453 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_917, %add_916), kwargs = {})
#   %mul_1051 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_453, %clamp_max_75), kwargs = {})
#   %add_918 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_916, %mul_1051), kwargs = {})
#   %add_919 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_911, %add_918), kwargs = {})
#   %relu_263 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_919,), kwargs = {})
#   %le_13 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_263, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 16) % 16)
    x4 = ((xindex // 4) % 4)
    x3 = (xindex % 4)
    x5 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x3), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + (x3), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr15 + (x6), xmask)
    tmp51 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
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
    tmp31 = tl.full([XBLOCK], 2, tl.int32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tmp36 = tmp35 + tmp31
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tmp39 = tl.load(in_ptr12 + (tmp38 + 2*tmp34 + 4*x5), xmask, eviction_policy='evict_last')
    tmp41 = tmp40 + tmp31
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tl.load(in_ptr12 + (tmp43 + 2*tmp34 + 4*x5), xmask, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp39
    tmp47 = tmp45 * tmp46
    tmp48 = tmp39 + tmp47
    tmp50 = tmp29 + tmp49
    tmp52 = tmp51 + tmp31
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tmp55 = tl.load(in_ptr12 + (tmp38 + 2*tmp54 + 4*x5), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (tmp43 + 2*tmp54 + 4*x5), xmask, eviction_policy='evict_last')
    tmp57 = tmp56 - tmp55
    tmp58 = tmp57 * tmp46
    tmp59 = tmp55 + tmp58
    tmp60 = tmp59 - tmp48
    tmp62 = tmp60 * tmp61
    tmp63 = tmp48 + tmp62
    tmp64 = tmp50 + tmp63
    tmp65 = tl.full([1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tmp67 = 0.0
    tmp68 = tmp66 <= tmp67
    tl.store(in_out_ptr0 + (x6), tmp64, xmask)
    tl.store(out_ptr1 + (x6), tmp68, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuu6ur23jnvm4a5aboxedsumhmj5s56wlx46lwty4na7eblhqrfe.py
# Topologically Sorted Source Nodes: [input_179, input_184, y_59, input_186, y_60, y_61, relu_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_179 => add_925, mul_1059, mul_1060, sub_456
#   input_184 => add_929, mul_1065, mul_1066, sub_458
#   input_186 => add_932, mul_1068, mul_1069, sub_459
#   relu_267 => relu_267
#   y_59 => add_930
#   y_60 => add_933
#   y_61 => add_934
# Graph fragment:
#   %sub_456 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_301, %unsqueeze_2409), kwargs = {})
#   %mul_1059 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_456, %unsqueeze_2411), kwargs = {})
#   %mul_1060 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1059, %unsqueeze_2413), kwargs = {})
#   %add_925 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1060, %unsqueeze_2415), kwargs = {})
#   %sub_458 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_303, %unsqueeze_2425), kwargs = {})
#   %mul_1065 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_458, %unsqueeze_2427), kwargs = {})
#   %mul_1066 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1065, %unsqueeze_2429), kwargs = {})
#   %add_929 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1066, %unsqueeze_2431), kwargs = {})
#   %add_930 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_925, %add_929), kwargs = {})
#   %sub_459 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_304, %unsqueeze_2433), kwargs = {})
#   %mul_1068 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_459, %unsqueeze_2435), kwargs = {})
#   %mul_1069 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1068, %unsqueeze_2437), kwargs = {})
#   %add_932 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1069, %unsqueeze_2439), kwargs = {})
#   %add_933 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_930, %add_932), kwargs = {})
#   %add_934 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_933, %relu_259), kwargs = {})
#   %relu_267 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_934,), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_267, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr10 + (x3), xmask)
    tmp31 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr15 + (x3), xmask)
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp48 = 0.0
    tmp49 = tmp47 <= tmp48
    tl.store(in_out_ptr0 + (x3), tmp47, xmask)
    tl.store(out_ptr0 + (x3), tmp49, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6lxicw7tqjnd5kgwyiow5yblkvo55g74euyhbc4r5xau67lnsb2.py
# Topologically Sorted Source Nodes: [relu_261, x1], Original ATen: [aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   relu_261 => relu_261
#   x1 => _unsafe_index_124, _unsafe_index_125, add_937, mul_1072, sub_461
# Graph fragment:
#   %relu_261 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_903,), kwargs = {})
#   %_unsafe_index_124 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_261, [None, None, %convert_element_type_69, %convert_element_type_71]), kwargs = {})
#   %_unsafe_index_125 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_261, [None, None, %convert_element_type_69, %clamp_max_1]), kwargs = {})
#   %sub_461 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_125, %_unsafe_index_124), kwargs = {})
#   %mul_1072 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_461, %clamp_max_2), kwargs = {})
#   %add_937 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_124, %mul_1072), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_relu_sub_51 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_relu_sub_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_relu_sub_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_relu_sub_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tl.load(in_ptr2 + (tmp15 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp17 = triton_helpers.maximum(tmp10, tmp16)
    tmp18 = tmp17 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/jv/cjvspfsf32szzbh6g2nyuptkur77kngz5sw3k7fnn6geiqdaw324.py
# Topologically Sorted Source Nodes: [relu_263, x2], Original ATen: [aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   relu_263 => relu_263
#   x2 => _unsafe_index_128, _unsafe_index_129, add_942, mul_1077, sub_466
# Graph fragment:
#   %relu_263 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_919,), kwargs = {})
#   %_unsafe_index_128 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_263, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_129 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_263, [None, None, %convert_element_type_133, %clamp_max_9]), kwargs = {})
#   %sub_466 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_129, %_unsafe_index_128), kwargs = {})
#   %mul_1077 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_466, %clamp_max_10), kwargs = {})
#   %add_942 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_128, %mul_1077), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_relu_sub_52 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_relu_sub_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_relu_sub_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_relu_sub_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tl.load(in_ptr2 + (tmp15 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp17 = triton_helpers.maximum(tmp10, tmp16)
    tmp18 = tmp17 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/7r/c7rgvssb6yqmzz6ydkjnbbmoe6xegriib3fqqa5sxo4xy6tjmlq3.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x3 => _unsafe_index_132, _unsafe_index_133, add_947, mul_1082, sub_471
# Graph fragment:
#   %_unsafe_index_132 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_267, [None, None, %convert_element_type_453, %convert_element_type_455]), kwargs = {})
#   %_unsafe_index_133 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_267, [None, None, %convert_element_type_453, %clamp_max_61]), kwargs = {})
#   %sub_471 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_133, %_unsafe_index_132), kwargs = {})
#   %mul_1082 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_471, %clamp_max_62), kwargs = {})
#   %add_947 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_132, %mul_1082), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_53 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/dn/cdnfznwtuzeamxggmyztrg4bc3ty726iajkfbbfbo7wrngmxuusb.py
# Topologically Sorted Source Nodes: [feats], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   feats => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_260, %add_939, %add_944, %add_949], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*i64', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 19, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 60)
    x3 = xindex // 15360
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 1024*x3), tmp4, other=0.0)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 12, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + (x4 + 256*((-4) + x2) + 2048*x3), tmp13, other=0.0)
    tmp15 = tl.load(in_ptr2 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full([XBLOCK], 8, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tmp20 = tl.load(in_ptr3 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp16
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr4 + (tmp23 + 8*tmp19 + 64*((-4) + x2) + 512*x3), tmp13, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.load(in_ptr5 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp27 + tmp16
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr4 + (tmp30 + 8*tmp19 + 64*((-4) + x2) + 512*x3), tmp13, eviction_policy='evict_last', other=0.0)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp32 - tmp26
    tmp34 = tl.load(in_ptr6 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp26 + tmp35
    tmp37 = tmp36 - tmp14
    tmp38 = tl.load(in_ptr7 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp14 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp13, tmp40, tmp41)
    tmp43 = tmp0 >= tmp11
    tmp44 = tl.full([1], 28, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr8 + (x4 + 256*((-12) + x2) + 4096*x3), tmp46, other=0.0)
    tmp48 = tl.load(in_ptr9 + (x1), tmp46, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full([XBLOCK], 4, tl.int32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp48 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp48)
    tmp53 = tl.load(in_ptr10 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp54 = tmp53 + tmp49
    tmp55 = tmp53 < 0
    tmp56 = tl.where(tmp55, tmp54, tmp53)
    tmp57 = tl.load(in_ptr11 + (tmp56 + 4*tmp52 + 16*((-12) + x2) + 256*x3), tmp46, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full([1], 0, tl.int32)
    tmp59 = triton_helpers.maximum(tmp58, tmp57)
    tmp60 = tl.load(in_ptr12 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 + tmp49
    tmp62 = tmp60 < 0
    tmp63 = tl.where(tmp62, tmp61, tmp60)
    tmp64 = tl.load(in_ptr11 + (tmp63 + 4*tmp52 + 16*((-12) + x2) + 256*x3), tmp46, eviction_policy='evict_last', other=0.0)
    tmp65 = triton_helpers.maximum(tmp58, tmp64)
    tmp66 = tmp65 - tmp59
    tmp67 = tl.load(in_ptr13 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 * tmp67
    tmp69 = tmp59 + tmp68
    tmp70 = tmp69 - tmp47
    tmp71 = tl.load(in_ptr14 + (x1), tmp46, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 * tmp71
    tmp73 = tmp47 + tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp46, tmp73, tmp74)
    tmp76 = tmp0 >= tmp44
    tmp77 = tl.full([1], 60, tl.int64)
    tmp78 = tmp0 < tmp77
    tmp79 = tl.load(in_ptr15 + (x4 + 256*((-28) + x2) + 8192*x3), tmp76, other=0.0)
    tmp80 = tl.load(in_ptr16 + (x1), tmp76, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.full([XBLOCK], 2, tl.int32)
    tmp82 = tmp80 + tmp81
    tmp83 = tmp80 < 0
    tmp84 = tl.where(tmp83, tmp82, tmp80)
    tmp85 = tl.load(in_ptr17 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp85 + tmp81
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tmp89 = tl.load(in_ptr18 + (tmp88 + 2*tmp84 + 4*((-28) + x2) + 128*x3), tmp76, eviction_policy='evict_last', other=0.0)
    tmp90 = tl.load(in_ptr19 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp81
    tmp92 = tmp90 < 0
    tmp93 = tl.where(tmp92, tmp91, tmp90)
    tmp94 = tl.load(in_ptr18 + (tmp93 + 2*tmp84 + 4*((-28) + x2) + 128*x3), tmp76, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp94 - tmp89
    tmp96 = tl.load(in_ptr20 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp97 = tmp95 * tmp96
    tmp98 = tmp89 + tmp97
    tmp99 = tmp98 - tmp79
    tmp100 = tl.load(in_ptr21 + (x1), tmp76, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp99 * tmp100
    tmp102 = tmp79 + tmp101
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp76, tmp102, tmp103)
    tmp105 = tl.where(tmp46, tmp75, tmp104)
    tmp106 = tl.where(tmp13, tmp42, tmp105)
    tmp107 = tl.where(tmp4, tmp9, tmp106)
    tl.store(out_ptr0 + (x5), tmp107, None)
''', device_str='cuda')


# kernel path: inductor_cache/or/cordv23w6pcjyg7zwlbu75r54vu7e4vzuhgyjirxby2zv32uszas.py
# Topologically Sorted Source Nodes: [input_187, input_188, input_189], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_187 => convolution_305
#   input_188 => add_951, mul_1086, mul_1087, sub_475
#   input_189 => relu_268
# Graph fragment:
#   %convolution_305 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_1527, %primals_1528, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_475 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_305, %unsqueeze_2441), kwargs = {})
#   %mul_1086 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_475, %unsqueeze_2443), kwargs = {})
#   %mul_1087 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1086, %unsqueeze_2445), kwargs = {})
#   %add_951 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1087, %unsqueeze_2447), kwargs = {})
#   %relu_268 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_951,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 60)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/cz/cczea36a6msoqgwe4n754imz2a72iuhtsxefbfkwehw5ycrg66mi.py
# Topologically Sorted Source Nodes: [input_190, probs_1], Original ATen: [aten.convolution, aten._softmax]
# Source node to ATen node mapping:
#   input_190 => convolution_306
#   probs_1 => div, exp, sum_1
# Graph fragment:
#   %convolution_306 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_268, %primals_1533, %primals_1534, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [2], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_1), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_1, 1), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_3,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [2], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_convolution_56 = async_compile.triton('triton_per_fused__softmax_convolution_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_convolution_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_convolution_56(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp8 = tmp4 - tmp7
    tmp9 = tmp8 * tmp3
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp10 / tmp13
    tl.store(in_out_ptr0 + (r2 + 256*x3), tmp2, None)
    tl.store(out_ptr2 + (r2 + 256*x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxgobdloqrvtc5tpuiqg57aq3w2iifkseffy57wuwcgex3ozsbm.py
# Topologically Sorted Source Nodes: [input_191, input_192, input_193], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_191 => convolution_307
#   input_192 => add_953, mul_1089, mul_1090, sub_476
#   input_193 => relu_269
# Graph fragment:
#   %convolution_307 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_1535, %primals_1536, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_476 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_307, %unsqueeze_2449), kwargs = {})
#   %mul_1089 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_476, %unsqueeze_2451), kwargs = {})
#   %mul_1090 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1089, %unsqueeze_2453), kwargs = {})
#   %add_953 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1090, %unsqueeze_2455), kwargs = {})
#   %relu_269 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_953,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 512)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvwh3pudqommny5nxf4cam3o7nsa6jyakq7wbtw773zfbv4h54j.py
# Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_195 => add_955, mul_1093, mul_1094, sub_478
#   input_196 => relu_270
# Graph fragment:
#   %sub_478 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_308, %unsqueeze_2458), kwargs = {})
#   %mul_1093 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_478, %unsqueeze_2460), kwargs = {})
#   %mul_1094 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1093, %unsqueeze_2462), kwargs = {})
#   %add_955 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1094, %unsqueeze_2464), kwargs = {})
#   %relu_270 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_955,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/sx/csx5hf7ikl7tj4rp4actu5grbuamppezjb2gtagfxcvkyj6gg57b.py
# Topologically Sorted Source Nodes: [input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_198 => add_957, mul_1096, mul_1097, sub_479
#   input_199 => relu_271
# Graph fragment:
#   %sub_479 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_309, %unsqueeze_2466), kwargs = {})
#   %mul_1096 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_479, %unsqueeze_2468), kwargs = {})
#   %mul_1097 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1096, %unsqueeze_2470), kwargs = {})
#   %add_957 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1097, %unsqueeze_2472), kwargs = {})
#   %relu_271 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_957,), kwargs = {})
#   %le_5 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_271, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clfcgfzrbi6fahqmeq7wveblsgutgsyqr6qksyelrrjzgmwc3cza.py
# Topologically Sorted Source Nodes: [input_200, input_206], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_200 => convolution_310
#   input_206 => convolution_312
# Graph fragment:
#   %convolution_310 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_2456, %primals_1551, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_312 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_2456, %primals_1561, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_60 = async_compile.triton('triton_poi_fused_convolution_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_60(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 2048*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask)
    tl.store(out_ptr1 + (x2 + 4*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4s/c4sava22lhnxrehbaqw3ajtrfb2u4xti4clkqvgy2ylrmot5tgvy.py
# Topologically Sorted Source Nodes: [input_201, input_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_201 => add_959, mul_1099, mul_1100, sub_480
#   input_202 => relu_272
# Graph fragment:
#   %sub_480 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_310, %unsqueeze_2474), kwargs = {})
#   %mul_1099 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_480, %unsqueeze_2476), kwargs = {})
#   %mul_1100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1099, %unsqueeze_2478), kwargs = {})
#   %add_959 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1100, %unsqueeze_2480), kwargs = {})
#   %relu_272 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_959,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
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


# kernel path: inductor_cache/4f/c4f2gkqclt5pk2dglnp7splj4dajop7pg6ye447aggqnty54f72z.py
# Topologically Sorted Source Nodes: [input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_204 => add_961, mul_1102, mul_1103, sub_481
#   input_205 => relu_273
# Graph fragment:
#   %sub_481 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_311, %unsqueeze_2482), kwargs = {})
#   %mul_1102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_481, %unsqueeze_2484), kwargs = {})
#   %mul_1103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1102, %unsqueeze_2486), kwargs = {})
#   %add_961 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1103, %unsqueeze_2488), kwargs = {})
#   %relu_273 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_961,), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_273, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coe36de2u6qkwce26swqpwbrbpkvq7c7dt5xcm6r76avoon52ljq.py
# Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_2 => exp_1
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_1, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 0.0625), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_1,), kwargs = {})
triton_poi_fused__softmax_63 = async_compile.triton('triton_poi_fused__softmax_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_63(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr0 + (4*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 0.0625
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ry/cryh4oho2nfzpgcakf4dbcrnlbffu754an4dsgzkkdfocpzghhqh.py
# Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_2 => div_1, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
triton_poi_fused__softmax_64 = async_compile.triton('triton_poi_fused__softmax_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (4*x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg6phzsybr7f65ope4zwedlruo5tpwz3luapicbfgl7u2yl2jdft.py
# Topologically Sorted Source Nodes: [context_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   context_1 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_65 = async_compile.triton('triton_poi_fused_clone_65', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_65(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 256*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2sazu5wlrjvbhwthtvw36s27thj5mkhbbsaudgrdepqghpaitmo.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_275, %relu_269], 1), kwargs = {})
triton_poi_fused_cat_66 = async_compile.triton('triton_poi_fused_cat_66', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 1024)
    x0 = (xindex % 256)
    x2 = xindex // 262144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 131072*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
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
    tmp26 = tl.full([1], 1024, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 256*((-512) + x1) + 131072*x2), tmp25, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/m2/cm2wlvshru5gsxgfahc664dohoj4wsjfhvkwnjgg4o5iopjfwu2n.py
# Topologically Sorted Source Nodes: [input_213, input_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_213 => add_967, mul_1112, mul_1113, sub_485
#   input_214 => relu_276
# Graph fragment:
#   %sub_485 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_314, %unsqueeze_2506), kwargs = {})
#   %mul_1112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_485, %unsqueeze_2508), kwargs = {})
#   %mul_1113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1112, %unsqueeze_2510), kwargs = {})
#   %add_967 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1113, %unsqueeze_2512), kwargs = {})
#   %relu_276 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_967,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_67', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 512)
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


# kernel path: inductor_cache/au/cau5eahwfjwzlerllobw3437nhpv2hnvgb7mcg2bg4kyxgc3toeo.py
# Topologically Sorted Source Nodes: [out_768], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_768 => convolution_315
# Graph fragment:
#   %convolution_315 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_276, %primals_1576, %primals_1577, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_68 = async_compile.triton('triton_poi_fused_convolution_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_68(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463, primals_1464, primals_1465, primals_1466, primals_1467, primals_1468, primals_1469, primals_1470, primals_1471, primals_1472, primals_1473, primals_1474, primals_1475, primals_1476, primals_1477, primals_1478, primals_1479, primals_1480, primals_1481, primals_1482, primals_1483, primals_1484, primals_1485, primals_1486, primals_1487, primals_1488, primals_1489, primals_1490, primals_1491, primals_1492, primals_1493, primals_1494, primals_1495, primals_1496, primals_1497, primals_1498, primals_1499, primals_1500, primals_1501, primals_1502, primals_1503, primals_1504, primals_1505, primals_1506, primals_1507, primals_1508, primals_1509, primals_1510, primals_1511, primals_1512, primals_1513, primals_1514, primals_1515, primals_1516, primals_1517, primals_1518, primals_1519, primals_1520, primals_1521, primals_1522, primals_1523, primals_1524, primals_1525, primals_1526, primals_1527, primals_1528, primals_1529, primals_1530, primals_1531, primals_1532, primals_1533, primals_1534, primals_1535, primals_1536, primals_1537, primals_1538, primals_1539, primals_1540, primals_1541, primals_1542, primals_1543, primals_1544, primals_1545, primals_1546, primals_1547, primals_1548, primals_1549, primals_1550, primals_1551, primals_1552, primals_1553, primals_1554, primals_1555, primals_1556, primals_1557, primals_1558, primals_1559, primals_1560, primals_1561, primals_1562, primals_1563, primals_1564, primals_1565, primals_1566, primals_1567, primals_1568, primals_1569, primals_1570, primals_1571, primals_1572, primals_1573, primals_1574, primals_1575, primals_1576, primals_1577 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (4, ), (1, ))
    assert_size_stride(primals_81, (4, ), (1, ))
    assert_size_stride(primals_82, (8, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_83, (8, ), (1, ))
    assert_size_stride(primals_84, (8, ), (1, ))
    assert_size_stride(primals_85, (8, ), (1, ))
    assert_size_stride(primals_86, (8, ), (1, ))
    assert_size_stride(primals_87, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_88, (4, ), (1, ))
    assert_size_stride(primals_89, (4, ), (1, ))
    assert_size_stride(primals_90, (4, ), (1, ))
    assert_size_stride(primals_91, (4, ), (1, ))
    assert_size_stride(primals_92, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_93, (4, ), (1, ))
    assert_size_stride(primals_94, (4, ), (1, ))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (4, ), (1, ))
    assert_size_stride(primals_97, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_98, (4, ), (1, ))
    assert_size_stride(primals_99, (4, ), (1, ))
    assert_size_stride(primals_100, (4, ), (1, ))
    assert_size_stride(primals_101, (4, ), (1, ))
    assert_size_stride(primals_102, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_103, (4, ), (1, ))
    assert_size_stride(primals_104, (4, ), (1, ))
    assert_size_stride(primals_105, (4, ), (1, ))
    assert_size_stride(primals_106, (4, ), (1, ))
    assert_size_stride(primals_107, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_108, (4, ), (1, ))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (4, ), (1, ))
    assert_size_stride(primals_111, (4, ), (1, ))
    assert_size_stride(primals_112, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (4, ), (1, ))
    assert_size_stride(primals_115, (4, ), (1, ))
    assert_size_stride(primals_116, (4, ), (1, ))
    assert_size_stride(primals_117, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_118, (4, ), (1, ))
    assert_size_stride(primals_119, (4, ), (1, ))
    assert_size_stride(primals_120, (4, ), (1, ))
    assert_size_stride(primals_121, (4, ), (1, ))
    assert_size_stride(primals_122, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_123, (4, ), (1, ))
    assert_size_stride(primals_124, (4, ), (1, ))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (4, ), (1, ))
    assert_size_stride(primals_127, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_128, (8, ), (1, ))
    assert_size_stride(primals_129, (8, ), (1, ))
    assert_size_stride(primals_130, (8, ), (1, ))
    assert_size_stride(primals_131, (8, ), (1, ))
    assert_size_stride(primals_132, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_133, (8, ), (1, ))
    assert_size_stride(primals_134, (8, ), (1, ))
    assert_size_stride(primals_135, (8, ), (1, ))
    assert_size_stride(primals_136, (8, ), (1, ))
    assert_size_stride(primals_137, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_138, (8, ), (1, ))
    assert_size_stride(primals_139, (8, ), (1, ))
    assert_size_stride(primals_140, (8, ), (1, ))
    assert_size_stride(primals_141, (8, ), (1, ))
    assert_size_stride(primals_142, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_143, (8, ), (1, ))
    assert_size_stride(primals_144, (8, ), (1, ))
    assert_size_stride(primals_145, (8, ), (1, ))
    assert_size_stride(primals_146, (8, ), (1, ))
    assert_size_stride(primals_147, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_148, (8, ), (1, ))
    assert_size_stride(primals_149, (8, ), (1, ))
    assert_size_stride(primals_150, (8, ), (1, ))
    assert_size_stride(primals_151, (8, ), (1, ))
    assert_size_stride(primals_152, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_153, (8, ), (1, ))
    assert_size_stride(primals_154, (8, ), (1, ))
    assert_size_stride(primals_155, (8, ), (1, ))
    assert_size_stride(primals_156, (8, ), (1, ))
    assert_size_stride(primals_157, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_158, (8, ), (1, ))
    assert_size_stride(primals_159, (8, ), (1, ))
    assert_size_stride(primals_160, (8, ), (1, ))
    assert_size_stride(primals_161, (8, ), (1, ))
    assert_size_stride(primals_162, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_163, (8, ), (1, ))
    assert_size_stride(primals_164, (8, ), (1, ))
    assert_size_stride(primals_165, (8, ), (1, ))
    assert_size_stride(primals_166, (8, ), (1, ))
    assert_size_stride(primals_167, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_168, (4, ), (1, ))
    assert_size_stride(primals_169, (4, ), (1, ))
    assert_size_stride(primals_170, (4, ), (1, ))
    assert_size_stride(primals_171, (4, ), (1, ))
    assert_size_stride(primals_172, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_173, (8, ), (1, ))
    assert_size_stride(primals_174, (8, ), (1, ))
    assert_size_stride(primals_175, (8, ), (1, ))
    assert_size_stride(primals_176, (8, ), (1, ))
    assert_size_stride(primals_177, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (16, ), (1, ))
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_183, (4, ), (1, ))
    assert_size_stride(primals_184, (4, ), (1, ))
    assert_size_stride(primals_185, (4, ), (1, ))
    assert_size_stride(primals_186, (4, ), (1, ))
    assert_size_stride(primals_187, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_188, (4, ), (1, ))
    assert_size_stride(primals_189, (4, ), (1, ))
    assert_size_stride(primals_190, (4, ), (1, ))
    assert_size_stride(primals_191, (4, ), (1, ))
    assert_size_stride(primals_192, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_193, (4, ), (1, ))
    assert_size_stride(primals_194, (4, ), (1, ))
    assert_size_stride(primals_195, (4, ), (1, ))
    assert_size_stride(primals_196, (4, ), (1, ))
    assert_size_stride(primals_197, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_198, (4, ), (1, ))
    assert_size_stride(primals_199, (4, ), (1, ))
    assert_size_stride(primals_200, (4, ), (1, ))
    assert_size_stride(primals_201, (4, ), (1, ))
    assert_size_stride(primals_202, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_203, (4, ), (1, ))
    assert_size_stride(primals_204, (4, ), (1, ))
    assert_size_stride(primals_205, (4, ), (1, ))
    assert_size_stride(primals_206, (4, ), (1, ))
    assert_size_stride(primals_207, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_208, (4, ), (1, ))
    assert_size_stride(primals_209, (4, ), (1, ))
    assert_size_stride(primals_210, (4, ), (1, ))
    assert_size_stride(primals_211, (4, ), (1, ))
    assert_size_stride(primals_212, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_213, (4, ), (1, ))
    assert_size_stride(primals_214, (4, ), (1, ))
    assert_size_stride(primals_215, (4, ), (1, ))
    assert_size_stride(primals_216, (4, ), (1, ))
    assert_size_stride(primals_217, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_218, (4, ), (1, ))
    assert_size_stride(primals_219, (4, ), (1, ))
    assert_size_stride(primals_220, (4, ), (1, ))
    assert_size_stride(primals_221, (4, ), (1, ))
    assert_size_stride(primals_222, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_223, (8, ), (1, ))
    assert_size_stride(primals_224, (8, ), (1, ))
    assert_size_stride(primals_225, (8, ), (1, ))
    assert_size_stride(primals_226, (8, ), (1, ))
    assert_size_stride(primals_227, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_228, (8, ), (1, ))
    assert_size_stride(primals_229, (8, ), (1, ))
    assert_size_stride(primals_230, (8, ), (1, ))
    assert_size_stride(primals_231, (8, ), (1, ))
    assert_size_stride(primals_232, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_233, (8, ), (1, ))
    assert_size_stride(primals_234, (8, ), (1, ))
    assert_size_stride(primals_235, (8, ), (1, ))
    assert_size_stride(primals_236, (8, ), (1, ))
    assert_size_stride(primals_237, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_238, (8, ), (1, ))
    assert_size_stride(primals_239, (8, ), (1, ))
    assert_size_stride(primals_240, (8, ), (1, ))
    assert_size_stride(primals_241, (8, ), (1, ))
    assert_size_stride(primals_242, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_243, (8, ), (1, ))
    assert_size_stride(primals_244, (8, ), (1, ))
    assert_size_stride(primals_245, (8, ), (1, ))
    assert_size_stride(primals_246, (8, ), (1, ))
    assert_size_stride(primals_247, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_248, (8, ), (1, ))
    assert_size_stride(primals_249, (8, ), (1, ))
    assert_size_stride(primals_250, (8, ), (1, ))
    assert_size_stride(primals_251, (8, ), (1, ))
    assert_size_stride(primals_252, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_253, (8, ), (1, ))
    assert_size_stride(primals_254, (8, ), (1, ))
    assert_size_stride(primals_255, (8, ), (1, ))
    assert_size_stride(primals_256, (8, ), (1, ))
    assert_size_stride(primals_257, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_258, (8, ), (1, ))
    assert_size_stride(primals_259, (8, ), (1, ))
    assert_size_stride(primals_260, (8, ), (1, ))
    assert_size_stride(primals_261, (8, ), (1, ))
    assert_size_stride(primals_262, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_263, (16, ), (1, ))
    assert_size_stride(primals_264, (16, ), (1, ))
    assert_size_stride(primals_265, (16, ), (1, ))
    assert_size_stride(primals_266, (16, ), (1, ))
    assert_size_stride(primals_267, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_268, (16, ), (1, ))
    assert_size_stride(primals_269, (16, ), (1, ))
    assert_size_stride(primals_270, (16, ), (1, ))
    assert_size_stride(primals_271, (16, ), (1, ))
    assert_size_stride(primals_272, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_273, (16, ), (1, ))
    assert_size_stride(primals_274, (16, ), (1, ))
    assert_size_stride(primals_275, (16, ), (1, ))
    assert_size_stride(primals_276, (16, ), (1, ))
    assert_size_stride(primals_277, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_278, (16, ), (1, ))
    assert_size_stride(primals_279, (16, ), (1, ))
    assert_size_stride(primals_280, (16, ), (1, ))
    assert_size_stride(primals_281, (16, ), (1, ))
    assert_size_stride(primals_282, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_283, (16, ), (1, ))
    assert_size_stride(primals_284, (16, ), (1, ))
    assert_size_stride(primals_285, (16, ), (1, ))
    assert_size_stride(primals_286, (16, ), (1, ))
    assert_size_stride(primals_287, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_288, (16, ), (1, ))
    assert_size_stride(primals_289, (16, ), (1, ))
    assert_size_stride(primals_290, (16, ), (1, ))
    assert_size_stride(primals_291, (16, ), (1, ))
    assert_size_stride(primals_292, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_293, (16, ), (1, ))
    assert_size_stride(primals_294, (16, ), (1, ))
    assert_size_stride(primals_295, (16, ), (1, ))
    assert_size_stride(primals_296, (16, ), (1, ))
    assert_size_stride(primals_297, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_298, (16, ), (1, ))
    assert_size_stride(primals_299, (16, ), (1, ))
    assert_size_stride(primals_300, (16, ), (1, ))
    assert_size_stride(primals_301, (16, ), (1, ))
    assert_size_stride(primals_302, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_303, (4, ), (1, ))
    assert_size_stride(primals_304, (4, ), (1, ))
    assert_size_stride(primals_305, (4, ), (1, ))
    assert_size_stride(primals_306, (4, ), (1, ))
    assert_size_stride(primals_307, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_308, (4, ), (1, ))
    assert_size_stride(primals_309, (4, ), (1, ))
    assert_size_stride(primals_310, (4, ), (1, ))
    assert_size_stride(primals_311, (4, ), (1, ))
    assert_size_stride(primals_312, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_313, (8, ), (1, ))
    assert_size_stride(primals_314, (8, ), (1, ))
    assert_size_stride(primals_315, (8, ), (1, ))
    assert_size_stride(primals_316, (8, ), (1, ))
    assert_size_stride(primals_317, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_318, (8, ), (1, ))
    assert_size_stride(primals_319, (8, ), (1, ))
    assert_size_stride(primals_320, (8, ), (1, ))
    assert_size_stride(primals_321, (8, ), (1, ))
    assert_size_stride(primals_322, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_323, (4, ), (1, ))
    assert_size_stride(primals_324, (4, ), (1, ))
    assert_size_stride(primals_325, (4, ), (1, ))
    assert_size_stride(primals_326, (4, ), (1, ))
    assert_size_stride(primals_327, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_328, (16, ), (1, ))
    assert_size_stride(primals_329, (16, ), (1, ))
    assert_size_stride(primals_330, (16, ), (1, ))
    assert_size_stride(primals_331, (16, ), (1, ))
    assert_size_stride(primals_332, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_333, (16, ), (1, ))
    assert_size_stride(primals_334, (16, ), (1, ))
    assert_size_stride(primals_335, (16, ), (1, ))
    assert_size_stride(primals_336, (16, ), (1, ))
    assert_size_stride(primals_337, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_338, (4, ), (1, ))
    assert_size_stride(primals_339, (4, ), (1, ))
    assert_size_stride(primals_340, (4, ), (1, ))
    assert_size_stride(primals_341, (4, ), (1, ))
    assert_size_stride(primals_342, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_343, (4, ), (1, ))
    assert_size_stride(primals_344, (4, ), (1, ))
    assert_size_stride(primals_345, (4, ), (1, ))
    assert_size_stride(primals_346, (4, ), (1, ))
    assert_size_stride(primals_347, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_348, (4, ), (1, ))
    assert_size_stride(primals_349, (4, ), (1, ))
    assert_size_stride(primals_350, (4, ), (1, ))
    assert_size_stride(primals_351, (4, ), (1, ))
    assert_size_stride(primals_352, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_353, (4, ), (1, ))
    assert_size_stride(primals_354, (4, ), (1, ))
    assert_size_stride(primals_355, (4, ), (1, ))
    assert_size_stride(primals_356, (4, ), (1, ))
    assert_size_stride(primals_357, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_358, (4, ), (1, ))
    assert_size_stride(primals_359, (4, ), (1, ))
    assert_size_stride(primals_360, (4, ), (1, ))
    assert_size_stride(primals_361, (4, ), (1, ))
    assert_size_stride(primals_362, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_363, (4, ), (1, ))
    assert_size_stride(primals_364, (4, ), (1, ))
    assert_size_stride(primals_365, (4, ), (1, ))
    assert_size_stride(primals_366, (4, ), (1, ))
    assert_size_stride(primals_367, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_368, (4, ), (1, ))
    assert_size_stride(primals_369, (4, ), (1, ))
    assert_size_stride(primals_370, (4, ), (1, ))
    assert_size_stride(primals_371, (4, ), (1, ))
    assert_size_stride(primals_372, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_373, (4, ), (1, ))
    assert_size_stride(primals_374, (4, ), (1, ))
    assert_size_stride(primals_375, (4, ), (1, ))
    assert_size_stride(primals_376, (4, ), (1, ))
    assert_size_stride(primals_377, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_378, (8, ), (1, ))
    assert_size_stride(primals_379, (8, ), (1, ))
    assert_size_stride(primals_380, (8, ), (1, ))
    assert_size_stride(primals_381, (8, ), (1, ))
    assert_size_stride(primals_382, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_383, (8, ), (1, ))
    assert_size_stride(primals_384, (8, ), (1, ))
    assert_size_stride(primals_385, (8, ), (1, ))
    assert_size_stride(primals_386, (8, ), (1, ))
    assert_size_stride(primals_387, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_388, (8, ), (1, ))
    assert_size_stride(primals_389, (8, ), (1, ))
    assert_size_stride(primals_390, (8, ), (1, ))
    assert_size_stride(primals_391, (8, ), (1, ))
    assert_size_stride(primals_392, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_393, (8, ), (1, ))
    assert_size_stride(primals_394, (8, ), (1, ))
    assert_size_stride(primals_395, (8, ), (1, ))
    assert_size_stride(primals_396, (8, ), (1, ))
    assert_size_stride(primals_397, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_398, (8, ), (1, ))
    assert_size_stride(primals_399, (8, ), (1, ))
    assert_size_stride(primals_400, (8, ), (1, ))
    assert_size_stride(primals_401, (8, ), (1, ))
    assert_size_stride(primals_402, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_403, (8, ), (1, ))
    assert_size_stride(primals_404, (8, ), (1, ))
    assert_size_stride(primals_405, (8, ), (1, ))
    assert_size_stride(primals_406, (8, ), (1, ))
    assert_size_stride(primals_407, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_408, (8, ), (1, ))
    assert_size_stride(primals_409, (8, ), (1, ))
    assert_size_stride(primals_410, (8, ), (1, ))
    assert_size_stride(primals_411, (8, ), (1, ))
    assert_size_stride(primals_412, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_413, (8, ), (1, ))
    assert_size_stride(primals_414, (8, ), (1, ))
    assert_size_stride(primals_415, (8, ), (1, ))
    assert_size_stride(primals_416, (8, ), (1, ))
    assert_size_stride(primals_417, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_418, (16, ), (1, ))
    assert_size_stride(primals_419, (16, ), (1, ))
    assert_size_stride(primals_420, (16, ), (1, ))
    assert_size_stride(primals_421, (16, ), (1, ))
    assert_size_stride(primals_422, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_423, (16, ), (1, ))
    assert_size_stride(primals_424, (16, ), (1, ))
    assert_size_stride(primals_425, (16, ), (1, ))
    assert_size_stride(primals_426, (16, ), (1, ))
    assert_size_stride(primals_427, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_428, (16, ), (1, ))
    assert_size_stride(primals_429, (16, ), (1, ))
    assert_size_stride(primals_430, (16, ), (1, ))
    assert_size_stride(primals_431, (16, ), (1, ))
    assert_size_stride(primals_432, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_433, (16, ), (1, ))
    assert_size_stride(primals_434, (16, ), (1, ))
    assert_size_stride(primals_435, (16, ), (1, ))
    assert_size_stride(primals_436, (16, ), (1, ))
    assert_size_stride(primals_437, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_438, (16, ), (1, ))
    assert_size_stride(primals_439, (16, ), (1, ))
    assert_size_stride(primals_440, (16, ), (1, ))
    assert_size_stride(primals_441, (16, ), (1, ))
    assert_size_stride(primals_442, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_443, (16, ), (1, ))
    assert_size_stride(primals_444, (16, ), (1, ))
    assert_size_stride(primals_445, (16, ), (1, ))
    assert_size_stride(primals_446, (16, ), (1, ))
    assert_size_stride(primals_447, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_448, (16, ), (1, ))
    assert_size_stride(primals_449, (16, ), (1, ))
    assert_size_stride(primals_450, (16, ), (1, ))
    assert_size_stride(primals_451, (16, ), (1, ))
    assert_size_stride(primals_452, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_453, (16, ), (1, ))
    assert_size_stride(primals_454, (16, ), (1, ))
    assert_size_stride(primals_455, (16, ), (1, ))
    assert_size_stride(primals_456, (16, ), (1, ))
    assert_size_stride(primals_457, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_458, (4, ), (1, ))
    assert_size_stride(primals_459, (4, ), (1, ))
    assert_size_stride(primals_460, (4, ), (1, ))
    assert_size_stride(primals_461, (4, ), (1, ))
    assert_size_stride(primals_462, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_463, (4, ), (1, ))
    assert_size_stride(primals_464, (4, ), (1, ))
    assert_size_stride(primals_465, (4, ), (1, ))
    assert_size_stride(primals_466, (4, ), (1, ))
    assert_size_stride(primals_467, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_468, (8, ), (1, ))
    assert_size_stride(primals_469, (8, ), (1, ))
    assert_size_stride(primals_470, (8, ), (1, ))
    assert_size_stride(primals_471, (8, ), (1, ))
    assert_size_stride(primals_472, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_473, (8, ), (1, ))
    assert_size_stride(primals_474, (8, ), (1, ))
    assert_size_stride(primals_475, (8, ), (1, ))
    assert_size_stride(primals_476, (8, ), (1, ))
    assert_size_stride(primals_477, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_478, (4, ), (1, ))
    assert_size_stride(primals_479, (4, ), (1, ))
    assert_size_stride(primals_480, (4, ), (1, ))
    assert_size_stride(primals_481, (4, ), (1, ))
    assert_size_stride(primals_482, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_483, (16, ), (1, ))
    assert_size_stride(primals_484, (16, ), (1, ))
    assert_size_stride(primals_485, (16, ), (1, ))
    assert_size_stride(primals_486, (16, ), (1, ))
    assert_size_stride(primals_487, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_488, (16, ), (1, ))
    assert_size_stride(primals_489, (16, ), (1, ))
    assert_size_stride(primals_490, (16, ), (1, ))
    assert_size_stride(primals_491, (16, ), (1, ))
    assert_size_stride(primals_492, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_493, (4, ), (1, ))
    assert_size_stride(primals_494, (4, ), (1, ))
    assert_size_stride(primals_495, (4, ), (1, ))
    assert_size_stride(primals_496, (4, ), (1, ))
    assert_size_stride(primals_497, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_498, (4, ), (1, ))
    assert_size_stride(primals_499, (4, ), (1, ))
    assert_size_stride(primals_500, (4, ), (1, ))
    assert_size_stride(primals_501, (4, ), (1, ))
    assert_size_stride(primals_502, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_503, (4, ), (1, ))
    assert_size_stride(primals_504, (4, ), (1, ))
    assert_size_stride(primals_505, (4, ), (1, ))
    assert_size_stride(primals_506, (4, ), (1, ))
    assert_size_stride(primals_507, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_508, (4, ), (1, ))
    assert_size_stride(primals_509, (4, ), (1, ))
    assert_size_stride(primals_510, (4, ), (1, ))
    assert_size_stride(primals_511, (4, ), (1, ))
    assert_size_stride(primals_512, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_513, (4, ), (1, ))
    assert_size_stride(primals_514, (4, ), (1, ))
    assert_size_stride(primals_515, (4, ), (1, ))
    assert_size_stride(primals_516, (4, ), (1, ))
    assert_size_stride(primals_517, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_518, (4, ), (1, ))
    assert_size_stride(primals_519, (4, ), (1, ))
    assert_size_stride(primals_520, (4, ), (1, ))
    assert_size_stride(primals_521, (4, ), (1, ))
    assert_size_stride(primals_522, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_523, (4, ), (1, ))
    assert_size_stride(primals_524, (4, ), (1, ))
    assert_size_stride(primals_525, (4, ), (1, ))
    assert_size_stride(primals_526, (4, ), (1, ))
    assert_size_stride(primals_527, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_528, (4, ), (1, ))
    assert_size_stride(primals_529, (4, ), (1, ))
    assert_size_stride(primals_530, (4, ), (1, ))
    assert_size_stride(primals_531, (4, ), (1, ))
    assert_size_stride(primals_532, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_533, (8, ), (1, ))
    assert_size_stride(primals_534, (8, ), (1, ))
    assert_size_stride(primals_535, (8, ), (1, ))
    assert_size_stride(primals_536, (8, ), (1, ))
    assert_size_stride(primals_537, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_538, (8, ), (1, ))
    assert_size_stride(primals_539, (8, ), (1, ))
    assert_size_stride(primals_540, (8, ), (1, ))
    assert_size_stride(primals_541, (8, ), (1, ))
    assert_size_stride(primals_542, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_543, (8, ), (1, ))
    assert_size_stride(primals_544, (8, ), (1, ))
    assert_size_stride(primals_545, (8, ), (1, ))
    assert_size_stride(primals_546, (8, ), (1, ))
    assert_size_stride(primals_547, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_548, (8, ), (1, ))
    assert_size_stride(primals_549, (8, ), (1, ))
    assert_size_stride(primals_550, (8, ), (1, ))
    assert_size_stride(primals_551, (8, ), (1, ))
    assert_size_stride(primals_552, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_553, (8, ), (1, ))
    assert_size_stride(primals_554, (8, ), (1, ))
    assert_size_stride(primals_555, (8, ), (1, ))
    assert_size_stride(primals_556, (8, ), (1, ))
    assert_size_stride(primals_557, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_558, (8, ), (1, ))
    assert_size_stride(primals_559, (8, ), (1, ))
    assert_size_stride(primals_560, (8, ), (1, ))
    assert_size_stride(primals_561, (8, ), (1, ))
    assert_size_stride(primals_562, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_563, (8, ), (1, ))
    assert_size_stride(primals_564, (8, ), (1, ))
    assert_size_stride(primals_565, (8, ), (1, ))
    assert_size_stride(primals_566, (8, ), (1, ))
    assert_size_stride(primals_567, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_568, (8, ), (1, ))
    assert_size_stride(primals_569, (8, ), (1, ))
    assert_size_stride(primals_570, (8, ), (1, ))
    assert_size_stride(primals_571, (8, ), (1, ))
    assert_size_stride(primals_572, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_573, (16, ), (1, ))
    assert_size_stride(primals_574, (16, ), (1, ))
    assert_size_stride(primals_575, (16, ), (1, ))
    assert_size_stride(primals_576, (16, ), (1, ))
    assert_size_stride(primals_577, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_578, (16, ), (1, ))
    assert_size_stride(primals_579, (16, ), (1, ))
    assert_size_stride(primals_580, (16, ), (1, ))
    assert_size_stride(primals_581, (16, ), (1, ))
    assert_size_stride(primals_582, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_583, (16, ), (1, ))
    assert_size_stride(primals_584, (16, ), (1, ))
    assert_size_stride(primals_585, (16, ), (1, ))
    assert_size_stride(primals_586, (16, ), (1, ))
    assert_size_stride(primals_587, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_588, (16, ), (1, ))
    assert_size_stride(primals_589, (16, ), (1, ))
    assert_size_stride(primals_590, (16, ), (1, ))
    assert_size_stride(primals_591, (16, ), (1, ))
    assert_size_stride(primals_592, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_593, (16, ), (1, ))
    assert_size_stride(primals_594, (16, ), (1, ))
    assert_size_stride(primals_595, (16, ), (1, ))
    assert_size_stride(primals_596, (16, ), (1, ))
    assert_size_stride(primals_597, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_598, (16, ), (1, ))
    assert_size_stride(primals_599, (16, ), (1, ))
    assert_size_stride(primals_600, (16, ), (1, ))
    assert_size_stride(primals_601, (16, ), (1, ))
    assert_size_stride(primals_602, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_603, (16, ), (1, ))
    assert_size_stride(primals_604, (16, ), (1, ))
    assert_size_stride(primals_605, (16, ), (1, ))
    assert_size_stride(primals_606, (16, ), (1, ))
    assert_size_stride(primals_607, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_608, (16, ), (1, ))
    assert_size_stride(primals_609, (16, ), (1, ))
    assert_size_stride(primals_610, (16, ), (1, ))
    assert_size_stride(primals_611, (16, ), (1, ))
    assert_size_stride(primals_612, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_613, (4, ), (1, ))
    assert_size_stride(primals_614, (4, ), (1, ))
    assert_size_stride(primals_615, (4, ), (1, ))
    assert_size_stride(primals_616, (4, ), (1, ))
    assert_size_stride(primals_617, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_618, (4, ), (1, ))
    assert_size_stride(primals_619, (4, ), (1, ))
    assert_size_stride(primals_620, (4, ), (1, ))
    assert_size_stride(primals_621, (4, ), (1, ))
    assert_size_stride(primals_622, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_623, (8, ), (1, ))
    assert_size_stride(primals_624, (8, ), (1, ))
    assert_size_stride(primals_625, (8, ), (1, ))
    assert_size_stride(primals_626, (8, ), (1, ))
    assert_size_stride(primals_627, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_628, (8, ), (1, ))
    assert_size_stride(primals_629, (8, ), (1, ))
    assert_size_stride(primals_630, (8, ), (1, ))
    assert_size_stride(primals_631, (8, ), (1, ))
    assert_size_stride(primals_632, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_633, (4, ), (1, ))
    assert_size_stride(primals_634, (4, ), (1, ))
    assert_size_stride(primals_635, (4, ), (1, ))
    assert_size_stride(primals_636, (4, ), (1, ))
    assert_size_stride(primals_637, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_638, (16, ), (1, ))
    assert_size_stride(primals_639, (16, ), (1, ))
    assert_size_stride(primals_640, (16, ), (1, ))
    assert_size_stride(primals_641, (16, ), (1, ))
    assert_size_stride(primals_642, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_643, (16, ), (1, ))
    assert_size_stride(primals_644, (16, ), (1, ))
    assert_size_stride(primals_645, (16, ), (1, ))
    assert_size_stride(primals_646, (16, ), (1, ))
    assert_size_stride(primals_647, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_648, (4, ), (1, ))
    assert_size_stride(primals_649, (4, ), (1, ))
    assert_size_stride(primals_650, (4, ), (1, ))
    assert_size_stride(primals_651, (4, ), (1, ))
    assert_size_stride(primals_652, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_653, (4, ), (1, ))
    assert_size_stride(primals_654, (4, ), (1, ))
    assert_size_stride(primals_655, (4, ), (1, ))
    assert_size_stride(primals_656, (4, ), (1, ))
    assert_size_stride(primals_657, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_658, (4, ), (1, ))
    assert_size_stride(primals_659, (4, ), (1, ))
    assert_size_stride(primals_660, (4, ), (1, ))
    assert_size_stride(primals_661, (4, ), (1, ))
    assert_size_stride(primals_662, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_663, (4, ), (1, ))
    assert_size_stride(primals_664, (4, ), (1, ))
    assert_size_stride(primals_665, (4, ), (1, ))
    assert_size_stride(primals_666, (4, ), (1, ))
    assert_size_stride(primals_667, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_668, (4, ), (1, ))
    assert_size_stride(primals_669, (4, ), (1, ))
    assert_size_stride(primals_670, (4, ), (1, ))
    assert_size_stride(primals_671, (4, ), (1, ))
    assert_size_stride(primals_672, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_673, (4, ), (1, ))
    assert_size_stride(primals_674, (4, ), (1, ))
    assert_size_stride(primals_675, (4, ), (1, ))
    assert_size_stride(primals_676, (4, ), (1, ))
    assert_size_stride(primals_677, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_678, (4, ), (1, ))
    assert_size_stride(primals_679, (4, ), (1, ))
    assert_size_stride(primals_680, (4, ), (1, ))
    assert_size_stride(primals_681, (4, ), (1, ))
    assert_size_stride(primals_682, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_683, (4, ), (1, ))
    assert_size_stride(primals_684, (4, ), (1, ))
    assert_size_stride(primals_685, (4, ), (1, ))
    assert_size_stride(primals_686, (4, ), (1, ))
    assert_size_stride(primals_687, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_688, (8, ), (1, ))
    assert_size_stride(primals_689, (8, ), (1, ))
    assert_size_stride(primals_690, (8, ), (1, ))
    assert_size_stride(primals_691, (8, ), (1, ))
    assert_size_stride(primals_692, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_693, (8, ), (1, ))
    assert_size_stride(primals_694, (8, ), (1, ))
    assert_size_stride(primals_695, (8, ), (1, ))
    assert_size_stride(primals_696, (8, ), (1, ))
    assert_size_stride(primals_697, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_698, (8, ), (1, ))
    assert_size_stride(primals_699, (8, ), (1, ))
    assert_size_stride(primals_700, (8, ), (1, ))
    assert_size_stride(primals_701, (8, ), (1, ))
    assert_size_stride(primals_702, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_703, (8, ), (1, ))
    assert_size_stride(primals_704, (8, ), (1, ))
    assert_size_stride(primals_705, (8, ), (1, ))
    assert_size_stride(primals_706, (8, ), (1, ))
    assert_size_stride(primals_707, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_708, (8, ), (1, ))
    assert_size_stride(primals_709, (8, ), (1, ))
    assert_size_stride(primals_710, (8, ), (1, ))
    assert_size_stride(primals_711, (8, ), (1, ))
    assert_size_stride(primals_712, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_713, (8, ), (1, ))
    assert_size_stride(primals_714, (8, ), (1, ))
    assert_size_stride(primals_715, (8, ), (1, ))
    assert_size_stride(primals_716, (8, ), (1, ))
    assert_size_stride(primals_717, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_718, (8, ), (1, ))
    assert_size_stride(primals_719, (8, ), (1, ))
    assert_size_stride(primals_720, (8, ), (1, ))
    assert_size_stride(primals_721, (8, ), (1, ))
    assert_size_stride(primals_722, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_723, (8, ), (1, ))
    assert_size_stride(primals_724, (8, ), (1, ))
    assert_size_stride(primals_725, (8, ), (1, ))
    assert_size_stride(primals_726, (8, ), (1, ))
    assert_size_stride(primals_727, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_728, (16, ), (1, ))
    assert_size_stride(primals_729, (16, ), (1, ))
    assert_size_stride(primals_730, (16, ), (1, ))
    assert_size_stride(primals_731, (16, ), (1, ))
    assert_size_stride(primals_732, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_733, (16, ), (1, ))
    assert_size_stride(primals_734, (16, ), (1, ))
    assert_size_stride(primals_735, (16, ), (1, ))
    assert_size_stride(primals_736, (16, ), (1, ))
    assert_size_stride(primals_737, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_738, (16, ), (1, ))
    assert_size_stride(primals_739, (16, ), (1, ))
    assert_size_stride(primals_740, (16, ), (1, ))
    assert_size_stride(primals_741, (16, ), (1, ))
    assert_size_stride(primals_742, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_743, (16, ), (1, ))
    assert_size_stride(primals_744, (16, ), (1, ))
    assert_size_stride(primals_745, (16, ), (1, ))
    assert_size_stride(primals_746, (16, ), (1, ))
    assert_size_stride(primals_747, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_748, (16, ), (1, ))
    assert_size_stride(primals_749, (16, ), (1, ))
    assert_size_stride(primals_750, (16, ), (1, ))
    assert_size_stride(primals_751, (16, ), (1, ))
    assert_size_stride(primals_752, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_753, (16, ), (1, ))
    assert_size_stride(primals_754, (16, ), (1, ))
    assert_size_stride(primals_755, (16, ), (1, ))
    assert_size_stride(primals_756, (16, ), (1, ))
    assert_size_stride(primals_757, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_758, (16, ), (1, ))
    assert_size_stride(primals_759, (16, ), (1, ))
    assert_size_stride(primals_760, (16, ), (1, ))
    assert_size_stride(primals_761, (16, ), (1, ))
    assert_size_stride(primals_762, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_763, (16, ), (1, ))
    assert_size_stride(primals_764, (16, ), (1, ))
    assert_size_stride(primals_765, (16, ), (1, ))
    assert_size_stride(primals_766, (16, ), (1, ))
    assert_size_stride(primals_767, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_768, (4, ), (1, ))
    assert_size_stride(primals_769, (4, ), (1, ))
    assert_size_stride(primals_770, (4, ), (1, ))
    assert_size_stride(primals_771, (4, ), (1, ))
    assert_size_stride(primals_772, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_773, (4, ), (1, ))
    assert_size_stride(primals_774, (4, ), (1, ))
    assert_size_stride(primals_775, (4, ), (1, ))
    assert_size_stride(primals_776, (4, ), (1, ))
    assert_size_stride(primals_777, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_778, (8, ), (1, ))
    assert_size_stride(primals_779, (8, ), (1, ))
    assert_size_stride(primals_780, (8, ), (1, ))
    assert_size_stride(primals_781, (8, ), (1, ))
    assert_size_stride(primals_782, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_783, (8, ), (1, ))
    assert_size_stride(primals_784, (8, ), (1, ))
    assert_size_stride(primals_785, (8, ), (1, ))
    assert_size_stride(primals_786, (8, ), (1, ))
    assert_size_stride(primals_787, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_788, (4, ), (1, ))
    assert_size_stride(primals_789, (4, ), (1, ))
    assert_size_stride(primals_790, (4, ), (1, ))
    assert_size_stride(primals_791, (4, ), (1, ))
    assert_size_stride(primals_792, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_793, (16, ), (1, ))
    assert_size_stride(primals_794, (16, ), (1, ))
    assert_size_stride(primals_795, (16, ), (1, ))
    assert_size_stride(primals_796, (16, ), (1, ))
    assert_size_stride(primals_797, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_798, (16, ), (1, ))
    assert_size_stride(primals_799, (16, ), (1, ))
    assert_size_stride(primals_800, (16, ), (1, ))
    assert_size_stride(primals_801, (16, ), (1, ))
    assert_size_stride(primals_802, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_803, (32, ), (1, ))
    assert_size_stride(primals_804, (32, ), (1, ))
    assert_size_stride(primals_805, (32, ), (1, ))
    assert_size_stride(primals_806, (32, ), (1, ))
    assert_size_stride(primals_807, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_808, (4, ), (1, ))
    assert_size_stride(primals_809, (4, ), (1, ))
    assert_size_stride(primals_810, (4, ), (1, ))
    assert_size_stride(primals_811, (4, ), (1, ))
    assert_size_stride(primals_812, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_813, (4, ), (1, ))
    assert_size_stride(primals_814, (4, ), (1, ))
    assert_size_stride(primals_815, (4, ), (1, ))
    assert_size_stride(primals_816, (4, ), (1, ))
    assert_size_stride(primals_817, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_818, (4, ), (1, ))
    assert_size_stride(primals_819, (4, ), (1, ))
    assert_size_stride(primals_820, (4, ), (1, ))
    assert_size_stride(primals_821, (4, ), (1, ))
    assert_size_stride(primals_822, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_823, (4, ), (1, ))
    assert_size_stride(primals_824, (4, ), (1, ))
    assert_size_stride(primals_825, (4, ), (1, ))
    assert_size_stride(primals_826, (4, ), (1, ))
    assert_size_stride(primals_827, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_828, (4, ), (1, ))
    assert_size_stride(primals_829, (4, ), (1, ))
    assert_size_stride(primals_830, (4, ), (1, ))
    assert_size_stride(primals_831, (4, ), (1, ))
    assert_size_stride(primals_832, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_833, (4, ), (1, ))
    assert_size_stride(primals_834, (4, ), (1, ))
    assert_size_stride(primals_835, (4, ), (1, ))
    assert_size_stride(primals_836, (4, ), (1, ))
    assert_size_stride(primals_837, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_838, (4, ), (1, ))
    assert_size_stride(primals_839, (4, ), (1, ))
    assert_size_stride(primals_840, (4, ), (1, ))
    assert_size_stride(primals_841, (4, ), (1, ))
    assert_size_stride(primals_842, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_843, (4, ), (1, ))
    assert_size_stride(primals_844, (4, ), (1, ))
    assert_size_stride(primals_845, (4, ), (1, ))
    assert_size_stride(primals_846, (4, ), (1, ))
    assert_size_stride(primals_847, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_848, (8, ), (1, ))
    assert_size_stride(primals_849, (8, ), (1, ))
    assert_size_stride(primals_850, (8, ), (1, ))
    assert_size_stride(primals_851, (8, ), (1, ))
    assert_size_stride(primals_852, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_853, (8, ), (1, ))
    assert_size_stride(primals_854, (8, ), (1, ))
    assert_size_stride(primals_855, (8, ), (1, ))
    assert_size_stride(primals_856, (8, ), (1, ))
    assert_size_stride(primals_857, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_858, (8, ), (1, ))
    assert_size_stride(primals_859, (8, ), (1, ))
    assert_size_stride(primals_860, (8, ), (1, ))
    assert_size_stride(primals_861, (8, ), (1, ))
    assert_size_stride(primals_862, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_863, (8, ), (1, ))
    assert_size_stride(primals_864, (8, ), (1, ))
    assert_size_stride(primals_865, (8, ), (1, ))
    assert_size_stride(primals_866, (8, ), (1, ))
    assert_size_stride(primals_867, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_868, (8, ), (1, ))
    assert_size_stride(primals_869, (8, ), (1, ))
    assert_size_stride(primals_870, (8, ), (1, ))
    assert_size_stride(primals_871, (8, ), (1, ))
    assert_size_stride(primals_872, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_873, (8, ), (1, ))
    assert_size_stride(primals_874, (8, ), (1, ))
    assert_size_stride(primals_875, (8, ), (1, ))
    assert_size_stride(primals_876, (8, ), (1, ))
    assert_size_stride(primals_877, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_878, (8, ), (1, ))
    assert_size_stride(primals_879, (8, ), (1, ))
    assert_size_stride(primals_880, (8, ), (1, ))
    assert_size_stride(primals_881, (8, ), (1, ))
    assert_size_stride(primals_882, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_883, (8, ), (1, ))
    assert_size_stride(primals_884, (8, ), (1, ))
    assert_size_stride(primals_885, (8, ), (1, ))
    assert_size_stride(primals_886, (8, ), (1, ))
    assert_size_stride(primals_887, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_888, (16, ), (1, ))
    assert_size_stride(primals_889, (16, ), (1, ))
    assert_size_stride(primals_890, (16, ), (1, ))
    assert_size_stride(primals_891, (16, ), (1, ))
    assert_size_stride(primals_892, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_893, (16, ), (1, ))
    assert_size_stride(primals_894, (16, ), (1, ))
    assert_size_stride(primals_895, (16, ), (1, ))
    assert_size_stride(primals_896, (16, ), (1, ))
    assert_size_stride(primals_897, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_898, (16, ), (1, ))
    assert_size_stride(primals_899, (16, ), (1, ))
    assert_size_stride(primals_900, (16, ), (1, ))
    assert_size_stride(primals_901, (16, ), (1, ))
    assert_size_stride(primals_902, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_903, (16, ), (1, ))
    assert_size_stride(primals_904, (16, ), (1, ))
    assert_size_stride(primals_905, (16, ), (1, ))
    assert_size_stride(primals_906, (16, ), (1, ))
    assert_size_stride(primals_907, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_908, (16, ), (1, ))
    assert_size_stride(primals_909, (16, ), (1, ))
    assert_size_stride(primals_910, (16, ), (1, ))
    assert_size_stride(primals_911, (16, ), (1, ))
    assert_size_stride(primals_912, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_913, (16, ), (1, ))
    assert_size_stride(primals_914, (16, ), (1, ))
    assert_size_stride(primals_915, (16, ), (1, ))
    assert_size_stride(primals_916, (16, ), (1, ))
    assert_size_stride(primals_917, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_918, (16, ), (1, ))
    assert_size_stride(primals_919, (16, ), (1, ))
    assert_size_stride(primals_920, (16, ), (1, ))
    assert_size_stride(primals_921, (16, ), (1, ))
    assert_size_stride(primals_922, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_923, (16, ), (1, ))
    assert_size_stride(primals_924, (16, ), (1, ))
    assert_size_stride(primals_925, (16, ), (1, ))
    assert_size_stride(primals_926, (16, ), (1, ))
    assert_size_stride(primals_927, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_928, (32, ), (1, ))
    assert_size_stride(primals_929, (32, ), (1, ))
    assert_size_stride(primals_930, (32, ), (1, ))
    assert_size_stride(primals_931, (32, ), (1, ))
    assert_size_stride(primals_932, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_933, (32, ), (1, ))
    assert_size_stride(primals_934, (32, ), (1, ))
    assert_size_stride(primals_935, (32, ), (1, ))
    assert_size_stride(primals_936, (32, ), (1, ))
    assert_size_stride(primals_937, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_938, (32, ), (1, ))
    assert_size_stride(primals_939, (32, ), (1, ))
    assert_size_stride(primals_940, (32, ), (1, ))
    assert_size_stride(primals_941, (32, ), (1, ))
    assert_size_stride(primals_942, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_943, (32, ), (1, ))
    assert_size_stride(primals_944, (32, ), (1, ))
    assert_size_stride(primals_945, (32, ), (1, ))
    assert_size_stride(primals_946, (32, ), (1, ))
    assert_size_stride(primals_947, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_948, (32, ), (1, ))
    assert_size_stride(primals_949, (32, ), (1, ))
    assert_size_stride(primals_950, (32, ), (1, ))
    assert_size_stride(primals_951, (32, ), (1, ))
    assert_size_stride(primals_952, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_953, (32, ), (1, ))
    assert_size_stride(primals_954, (32, ), (1, ))
    assert_size_stride(primals_955, (32, ), (1, ))
    assert_size_stride(primals_956, (32, ), (1, ))
    assert_size_stride(primals_957, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_958, (32, ), (1, ))
    assert_size_stride(primals_959, (32, ), (1, ))
    assert_size_stride(primals_960, (32, ), (1, ))
    assert_size_stride(primals_961, (32, ), (1, ))
    assert_size_stride(primals_962, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_963, (32, ), (1, ))
    assert_size_stride(primals_964, (32, ), (1, ))
    assert_size_stride(primals_965, (32, ), (1, ))
    assert_size_stride(primals_966, (32, ), (1, ))
    assert_size_stride(primals_967, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_968, (4, ), (1, ))
    assert_size_stride(primals_969, (4, ), (1, ))
    assert_size_stride(primals_970, (4, ), (1, ))
    assert_size_stride(primals_971, (4, ), (1, ))
    assert_size_stride(primals_972, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_973, (4, ), (1, ))
    assert_size_stride(primals_974, (4, ), (1, ))
    assert_size_stride(primals_975, (4, ), (1, ))
    assert_size_stride(primals_976, (4, ), (1, ))
    assert_size_stride(primals_977, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_978, (4, ), (1, ))
    assert_size_stride(primals_979, (4, ), (1, ))
    assert_size_stride(primals_980, (4, ), (1, ))
    assert_size_stride(primals_981, (4, ), (1, ))
    assert_size_stride(primals_982, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_983, (8, ), (1, ))
    assert_size_stride(primals_984, (8, ), (1, ))
    assert_size_stride(primals_985, (8, ), (1, ))
    assert_size_stride(primals_986, (8, ), (1, ))
    assert_size_stride(primals_987, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_988, (8, ), (1, ))
    assert_size_stride(primals_989, (8, ), (1, ))
    assert_size_stride(primals_990, (8, ), (1, ))
    assert_size_stride(primals_991, (8, ), (1, ))
    assert_size_stride(primals_992, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_993, (8, ), (1, ))
    assert_size_stride(primals_994, (8, ), (1, ))
    assert_size_stride(primals_995, (8, ), (1, ))
    assert_size_stride(primals_996, (8, ), (1, ))
    assert_size_stride(primals_997, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_998, (4, ), (1, ))
    assert_size_stride(primals_999, (4, ), (1, ))
    assert_size_stride(primals_1000, (4, ), (1, ))
    assert_size_stride(primals_1001, (4, ), (1, ))
    assert_size_stride(primals_1002, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1003, (16, ), (1, ))
    assert_size_stride(primals_1004, (16, ), (1, ))
    assert_size_stride(primals_1005, (16, ), (1, ))
    assert_size_stride(primals_1006, (16, ), (1, ))
    assert_size_stride(primals_1007, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1008, (16, ), (1, ))
    assert_size_stride(primals_1009, (16, ), (1, ))
    assert_size_stride(primals_1010, (16, ), (1, ))
    assert_size_stride(primals_1011, (16, ), (1, ))
    assert_size_stride(primals_1012, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1013, (16, ), (1, ))
    assert_size_stride(primals_1014, (16, ), (1, ))
    assert_size_stride(primals_1015, (16, ), (1, ))
    assert_size_stride(primals_1016, (16, ), (1, ))
    assert_size_stride(primals_1017, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1018, (4, ), (1, ))
    assert_size_stride(primals_1019, (4, ), (1, ))
    assert_size_stride(primals_1020, (4, ), (1, ))
    assert_size_stride(primals_1021, (4, ), (1, ))
    assert_size_stride(primals_1022, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1023, (4, ), (1, ))
    assert_size_stride(primals_1024, (4, ), (1, ))
    assert_size_stride(primals_1025, (4, ), (1, ))
    assert_size_stride(primals_1026, (4, ), (1, ))
    assert_size_stride(primals_1027, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1028, (32, ), (1, ))
    assert_size_stride(primals_1029, (32, ), (1, ))
    assert_size_stride(primals_1030, (32, ), (1, ))
    assert_size_stride(primals_1031, (32, ), (1, ))
    assert_size_stride(primals_1032, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1033, (8, ), (1, ))
    assert_size_stride(primals_1034, (8, ), (1, ))
    assert_size_stride(primals_1035, (8, ), (1, ))
    assert_size_stride(primals_1036, (8, ), (1, ))
    assert_size_stride(primals_1037, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1038, (32, ), (1, ))
    assert_size_stride(primals_1039, (32, ), (1, ))
    assert_size_stride(primals_1040, (32, ), (1, ))
    assert_size_stride(primals_1041, (32, ), (1, ))
    assert_size_stride(primals_1042, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1043, (32, ), (1, ))
    assert_size_stride(primals_1044, (32, ), (1, ))
    assert_size_stride(primals_1045, (32, ), (1, ))
    assert_size_stride(primals_1046, (32, ), (1, ))
    assert_size_stride(primals_1047, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1048, (4, ), (1, ))
    assert_size_stride(primals_1049, (4, ), (1, ))
    assert_size_stride(primals_1050, (4, ), (1, ))
    assert_size_stride(primals_1051, (4, ), (1, ))
    assert_size_stride(primals_1052, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1053, (4, ), (1, ))
    assert_size_stride(primals_1054, (4, ), (1, ))
    assert_size_stride(primals_1055, (4, ), (1, ))
    assert_size_stride(primals_1056, (4, ), (1, ))
    assert_size_stride(primals_1057, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1058, (4, ), (1, ))
    assert_size_stride(primals_1059, (4, ), (1, ))
    assert_size_stride(primals_1060, (4, ), (1, ))
    assert_size_stride(primals_1061, (4, ), (1, ))
    assert_size_stride(primals_1062, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1063, (4, ), (1, ))
    assert_size_stride(primals_1064, (4, ), (1, ))
    assert_size_stride(primals_1065, (4, ), (1, ))
    assert_size_stride(primals_1066, (4, ), (1, ))
    assert_size_stride(primals_1067, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1068, (4, ), (1, ))
    assert_size_stride(primals_1069, (4, ), (1, ))
    assert_size_stride(primals_1070, (4, ), (1, ))
    assert_size_stride(primals_1071, (4, ), (1, ))
    assert_size_stride(primals_1072, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1073, (4, ), (1, ))
    assert_size_stride(primals_1074, (4, ), (1, ))
    assert_size_stride(primals_1075, (4, ), (1, ))
    assert_size_stride(primals_1076, (4, ), (1, ))
    assert_size_stride(primals_1077, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1078, (4, ), (1, ))
    assert_size_stride(primals_1079, (4, ), (1, ))
    assert_size_stride(primals_1080, (4, ), (1, ))
    assert_size_stride(primals_1081, (4, ), (1, ))
    assert_size_stride(primals_1082, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1083, (4, ), (1, ))
    assert_size_stride(primals_1084, (4, ), (1, ))
    assert_size_stride(primals_1085, (4, ), (1, ))
    assert_size_stride(primals_1086, (4, ), (1, ))
    assert_size_stride(primals_1087, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1088, (8, ), (1, ))
    assert_size_stride(primals_1089, (8, ), (1, ))
    assert_size_stride(primals_1090, (8, ), (1, ))
    assert_size_stride(primals_1091, (8, ), (1, ))
    assert_size_stride(primals_1092, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1093, (8, ), (1, ))
    assert_size_stride(primals_1094, (8, ), (1, ))
    assert_size_stride(primals_1095, (8, ), (1, ))
    assert_size_stride(primals_1096, (8, ), (1, ))
    assert_size_stride(primals_1097, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1098, (8, ), (1, ))
    assert_size_stride(primals_1099, (8, ), (1, ))
    assert_size_stride(primals_1100, (8, ), (1, ))
    assert_size_stride(primals_1101, (8, ), (1, ))
    assert_size_stride(primals_1102, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1103, (8, ), (1, ))
    assert_size_stride(primals_1104, (8, ), (1, ))
    assert_size_stride(primals_1105, (8, ), (1, ))
    assert_size_stride(primals_1106, (8, ), (1, ))
    assert_size_stride(primals_1107, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1108, (8, ), (1, ))
    assert_size_stride(primals_1109, (8, ), (1, ))
    assert_size_stride(primals_1110, (8, ), (1, ))
    assert_size_stride(primals_1111, (8, ), (1, ))
    assert_size_stride(primals_1112, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1113, (8, ), (1, ))
    assert_size_stride(primals_1114, (8, ), (1, ))
    assert_size_stride(primals_1115, (8, ), (1, ))
    assert_size_stride(primals_1116, (8, ), (1, ))
    assert_size_stride(primals_1117, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1118, (8, ), (1, ))
    assert_size_stride(primals_1119, (8, ), (1, ))
    assert_size_stride(primals_1120, (8, ), (1, ))
    assert_size_stride(primals_1121, (8, ), (1, ))
    assert_size_stride(primals_1122, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1123, (8, ), (1, ))
    assert_size_stride(primals_1124, (8, ), (1, ))
    assert_size_stride(primals_1125, (8, ), (1, ))
    assert_size_stride(primals_1126, (8, ), (1, ))
    assert_size_stride(primals_1127, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1128, (16, ), (1, ))
    assert_size_stride(primals_1129, (16, ), (1, ))
    assert_size_stride(primals_1130, (16, ), (1, ))
    assert_size_stride(primals_1131, (16, ), (1, ))
    assert_size_stride(primals_1132, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1133, (16, ), (1, ))
    assert_size_stride(primals_1134, (16, ), (1, ))
    assert_size_stride(primals_1135, (16, ), (1, ))
    assert_size_stride(primals_1136, (16, ), (1, ))
    assert_size_stride(primals_1137, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1138, (16, ), (1, ))
    assert_size_stride(primals_1139, (16, ), (1, ))
    assert_size_stride(primals_1140, (16, ), (1, ))
    assert_size_stride(primals_1141, (16, ), (1, ))
    assert_size_stride(primals_1142, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1143, (16, ), (1, ))
    assert_size_stride(primals_1144, (16, ), (1, ))
    assert_size_stride(primals_1145, (16, ), (1, ))
    assert_size_stride(primals_1146, (16, ), (1, ))
    assert_size_stride(primals_1147, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1148, (16, ), (1, ))
    assert_size_stride(primals_1149, (16, ), (1, ))
    assert_size_stride(primals_1150, (16, ), (1, ))
    assert_size_stride(primals_1151, (16, ), (1, ))
    assert_size_stride(primals_1152, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1153, (16, ), (1, ))
    assert_size_stride(primals_1154, (16, ), (1, ))
    assert_size_stride(primals_1155, (16, ), (1, ))
    assert_size_stride(primals_1156, (16, ), (1, ))
    assert_size_stride(primals_1157, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1158, (16, ), (1, ))
    assert_size_stride(primals_1159, (16, ), (1, ))
    assert_size_stride(primals_1160, (16, ), (1, ))
    assert_size_stride(primals_1161, (16, ), (1, ))
    assert_size_stride(primals_1162, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1163, (16, ), (1, ))
    assert_size_stride(primals_1164, (16, ), (1, ))
    assert_size_stride(primals_1165, (16, ), (1, ))
    assert_size_stride(primals_1166, (16, ), (1, ))
    assert_size_stride(primals_1167, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1168, (32, ), (1, ))
    assert_size_stride(primals_1169, (32, ), (1, ))
    assert_size_stride(primals_1170, (32, ), (1, ))
    assert_size_stride(primals_1171, (32, ), (1, ))
    assert_size_stride(primals_1172, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1173, (32, ), (1, ))
    assert_size_stride(primals_1174, (32, ), (1, ))
    assert_size_stride(primals_1175, (32, ), (1, ))
    assert_size_stride(primals_1176, (32, ), (1, ))
    assert_size_stride(primals_1177, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1178, (32, ), (1, ))
    assert_size_stride(primals_1179, (32, ), (1, ))
    assert_size_stride(primals_1180, (32, ), (1, ))
    assert_size_stride(primals_1181, (32, ), (1, ))
    assert_size_stride(primals_1182, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1183, (32, ), (1, ))
    assert_size_stride(primals_1184, (32, ), (1, ))
    assert_size_stride(primals_1185, (32, ), (1, ))
    assert_size_stride(primals_1186, (32, ), (1, ))
    assert_size_stride(primals_1187, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1188, (32, ), (1, ))
    assert_size_stride(primals_1189, (32, ), (1, ))
    assert_size_stride(primals_1190, (32, ), (1, ))
    assert_size_stride(primals_1191, (32, ), (1, ))
    assert_size_stride(primals_1192, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1193, (32, ), (1, ))
    assert_size_stride(primals_1194, (32, ), (1, ))
    assert_size_stride(primals_1195, (32, ), (1, ))
    assert_size_stride(primals_1196, (32, ), (1, ))
    assert_size_stride(primals_1197, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1198, (32, ), (1, ))
    assert_size_stride(primals_1199, (32, ), (1, ))
    assert_size_stride(primals_1200, (32, ), (1, ))
    assert_size_stride(primals_1201, (32, ), (1, ))
    assert_size_stride(primals_1202, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1203, (32, ), (1, ))
    assert_size_stride(primals_1204, (32, ), (1, ))
    assert_size_stride(primals_1205, (32, ), (1, ))
    assert_size_stride(primals_1206, (32, ), (1, ))
    assert_size_stride(primals_1207, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_1208, (4, ), (1, ))
    assert_size_stride(primals_1209, (4, ), (1, ))
    assert_size_stride(primals_1210, (4, ), (1, ))
    assert_size_stride(primals_1211, (4, ), (1, ))
    assert_size_stride(primals_1212, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_1213, (4, ), (1, ))
    assert_size_stride(primals_1214, (4, ), (1, ))
    assert_size_stride(primals_1215, (4, ), (1, ))
    assert_size_stride(primals_1216, (4, ), (1, ))
    assert_size_stride(primals_1217, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1218, (4, ), (1, ))
    assert_size_stride(primals_1219, (4, ), (1, ))
    assert_size_stride(primals_1220, (4, ), (1, ))
    assert_size_stride(primals_1221, (4, ), (1, ))
    assert_size_stride(primals_1222, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1223, (8, ), (1, ))
    assert_size_stride(primals_1224, (8, ), (1, ))
    assert_size_stride(primals_1225, (8, ), (1, ))
    assert_size_stride(primals_1226, (8, ), (1, ))
    assert_size_stride(primals_1227, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_1228, (8, ), (1, ))
    assert_size_stride(primals_1229, (8, ), (1, ))
    assert_size_stride(primals_1230, (8, ), (1, ))
    assert_size_stride(primals_1231, (8, ), (1, ))
    assert_size_stride(primals_1232, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1233, (8, ), (1, ))
    assert_size_stride(primals_1234, (8, ), (1, ))
    assert_size_stride(primals_1235, (8, ), (1, ))
    assert_size_stride(primals_1236, (8, ), (1, ))
    assert_size_stride(primals_1237, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1238, (4, ), (1, ))
    assert_size_stride(primals_1239, (4, ), (1, ))
    assert_size_stride(primals_1240, (4, ), (1, ))
    assert_size_stride(primals_1241, (4, ), (1, ))
    assert_size_stride(primals_1242, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1243, (16, ), (1, ))
    assert_size_stride(primals_1244, (16, ), (1, ))
    assert_size_stride(primals_1245, (16, ), (1, ))
    assert_size_stride(primals_1246, (16, ), (1, ))
    assert_size_stride(primals_1247, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1248, (16, ), (1, ))
    assert_size_stride(primals_1249, (16, ), (1, ))
    assert_size_stride(primals_1250, (16, ), (1, ))
    assert_size_stride(primals_1251, (16, ), (1, ))
    assert_size_stride(primals_1252, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1253, (16, ), (1, ))
    assert_size_stride(primals_1254, (16, ), (1, ))
    assert_size_stride(primals_1255, (16, ), (1, ))
    assert_size_stride(primals_1256, (16, ), (1, ))
    assert_size_stride(primals_1257, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1258, (4, ), (1, ))
    assert_size_stride(primals_1259, (4, ), (1, ))
    assert_size_stride(primals_1260, (4, ), (1, ))
    assert_size_stride(primals_1261, (4, ), (1, ))
    assert_size_stride(primals_1262, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1263, (4, ), (1, ))
    assert_size_stride(primals_1264, (4, ), (1, ))
    assert_size_stride(primals_1265, (4, ), (1, ))
    assert_size_stride(primals_1266, (4, ), (1, ))
    assert_size_stride(primals_1267, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1268, (32, ), (1, ))
    assert_size_stride(primals_1269, (32, ), (1, ))
    assert_size_stride(primals_1270, (32, ), (1, ))
    assert_size_stride(primals_1271, (32, ), (1, ))
    assert_size_stride(primals_1272, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1273, (8, ), (1, ))
    assert_size_stride(primals_1274, (8, ), (1, ))
    assert_size_stride(primals_1275, (8, ), (1, ))
    assert_size_stride(primals_1276, (8, ), (1, ))
    assert_size_stride(primals_1277, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1278, (32, ), (1, ))
    assert_size_stride(primals_1279, (32, ), (1, ))
    assert_size_stride(primals_1280, (32, ), (1, ))
    assert_size_stride(primals_1281, (32, ), (1, ))
    assert_size_stride(primals_1282, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1283, (32, ), (1, ))
    assert_size_stride(primals_1284, (32, ), (1, ))
    assert_size_stride(primals_1285, (32, ), (1, ))
    assert_size_stride(primals_1286, (32, ), (1, ))
    assert_size_stride(primals_1287, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1288, (4, ), (1, ))
    assert_size_stride(primals_1289, (4, ), (1, ))
    assert_size_stride(primals_1290, (4, ), (1, ))
    assert_size_stride(primals_1291, (4, ), (1, ))
    assert_size_stride(primals_1292, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1293, (4, ), (1, ))
    assert_size_stride(primals_1294, (4, ), (1, ))
    assert_size_stride(primals_1295, (4, ), (1, ))
    assert_size_stride(primals_1296, (4, ), (1, ))
    assert_size_stride(primals_1297, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1298, (4, ), (1, ))
    assert_size_stride(primals_1299, (4, ), (1, ))
    assert_size_stride(primals_1300, (4, ), (1, ))
    assert_size_stride(primals_1301, (4, ), (1, ))
    assert_size_stride(primals_1302, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1303, (4, ), (1, ))
    assert_size_stride(primals_1304, (4, ), (1, ))
    assert_size_stride(primals_1305, (4, ), (1, ))
    assert_size_stride(primals_1306, (4, ), (1, ))
    assert_size_stride(primals_1307, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1308, (4, ), (1, ))
    assert_size_stride(primals_1309, (4, ), (1, ))
    assert_size_stride(primals_1310, (4, ), (1, ))
    assert_size_stride(primals_1311, (4, ), (1, ))
    assert_size_stride(primals_1312, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1313, (4, ), (1, ))
    assert_size_stride(primals_1314, (4, ), (1, ))
    assert_size_stride(primals_1315, (4, ), (1, ))
    assert_size_stride(primals_1316, (4, ), (1, ))
    assert_size_stride(primals_1317, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1318, (4, ), (1, ))
    assert_size_stride(primals_1319, (4, ), (1, ))
    assert_size_stride(primals_1320, (4, ), (1, ))
    assert_size_stride(primals_1321, (4, ), (1, ))
    assert_size_stride(primals_1322, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1323, (4, ), (1, ))
    assert_size_stride(primals_1324, (4, ), (1, ))
    assert_size_stride(primals_1325, (4, ), (1, ))
    assert_size_stride(primals_1326, (4, ), (1, ))
    assert_size_stride(primals_1327, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1328, (8, ), (1, ))
    assert_size_stride(primals_1329, (8, ), (1, ))
    assert_size_stride(primals_1330, (8, ), (1, ))
    assert_size_stride(primals_1331, (8, ), (1, ))
    assert_size_stride(primals_1332, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1333, (8, ), (1, ))
    assert_size_stride(primals_1334, (8, ), (1, ))
    assert_size_stride(primals_1335, (8, ), (1, ))
    assert_size_stride(primals_1336, (8, ), (1, ))
    assert_size_stride(primals_1337, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1338, (8, ), (1, ))
    assert_size_stride(primals_1339, (8, ), (1, ))
    assert_size_stride(primals_1340, (8, ), (1, ))
    assert_size_stride(primals_1341, (8, ), (1, ))
    assert_size_stride(primals_1342, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1343, (8, ), (1, ))
    assert_size_stride(primals_1344, (8, ), (1, ))
    assert_size_stride(primals_1345, (8, ), (1, ))
    assert_size_stride(primals_1346, (8, ), (1, ))
    assert_size_stride(primals_1347, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1348, (8, ), (1, ))
    assert_size_stride(primals_1349, (8, ), (1, ))
    assert_size_stride(primals_1350, (8, ), (1, ))
    assert_size_stride(primals_1351, (8, ), (1, ))
    assert_size_stride(primals_1352, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1353, (8, ), (1, ))
    assert_size_stride(primals_1354, (8, ), (1, ))
    assert_size_stride(primals_1355, (8, ), (1, ))
    assert_size_stride(primals_1356, (8, ), (1, ))
    assert_size_stride(primals_1357, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1358, (8, ), (1, ))
    assert_size_stride(primals_1359, (8, ), (1, ))
    assert_size_stride(primals_1360, (8, ), (1, ))
    assert_size_stride(primals_1361, (8, ), (1, ))
    assert_size_stride(primals_1362, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1363, (8, ), (1, ))
    assert_size_stride(primals_1364, (8, ), (1, ))
    assert_size_stride(primals_1365, (8, ), (1, ))
    assert_size_stride(primals_1366, (8, ), (1, ))
    assert_size_stride(primals_1367, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1368, (16, ), (1, ))
    assert_size_stride(primals_1369, (16, ), (1, ))
    assert_size_stride(primals_1370, (16, ), (1, ))
    assert_size_stride(primals_1371, (16, ), (1, ))
    assert_size_stride(primals_1372, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1373, (16, ), (1, ))
    assert_size_stride(primals_1374, (16, ), (1, ))
    assert_size_stride(primals_1375, (16, ), (1, ))
    assert_size_stride(primals_1376, (16, ), (1, ))
    assert_size_stride(primals_1377, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1378, (16, ), (1, ))
    assert_size_stride(primals_1379, (16, ), (1, ))
    assert_size_stride(primals_1380, (16, ), (1, ))
    assert_size_stride(primals_1381, (16, ), (1, ))
    assert_size_stride(primals_1382, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1383, (16, ), (1, ))
    assert_size_stride(primals_1384, (16, ), (1, ))
    assert_size_stride(primals_1385, (16, ), (1, ))
    assert_size_stride(primals_1386, (16, ), (1, ))
    assert_size_stride(primals_1387, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1388, (16, ), (1, ))
    assert_size_stride(primals_1389, (16, ), (1, ))
    assert_size_stride(primals_1390, (16, ), (1, ))
    assert_size_stride(primals_1391, (16, ), (1, ))
    assert_size_stride(primals_1392, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1393, (16, ), (1, ))
    assert_size_stride(primals_1394, (16, ), (1, ))
    assert_size_stride(primals_1395, (16, ), (1, ))
    assert_size_stride(primals_1396, (16, ), (1, ))
    assert_size_stride(primals_1397, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1398, (16, ), (1, ))
    assert_size_stride(primals_1399, (16, ), (1, ))
    assert_size_stride(primals_1400, (16, ), (1, ))
    assert_size_stride(primals_1401, (16, ), (1, ))
    assert_size_stride(primals_1402, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1403, (16, ), (1, ))
    assert_size_stride(primals_1404, (16, ), (1, ))
    assert_size_stride(primals_1405, (16, ), (1, ))
    assert_size_stride(primals_1406, (16, ), (1, ))
    assert_size_stride(primals_1407, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1408, (32, ), (1, ))
    assert_size_stride(primals_1409, (32, ), (1, ))
    assert_size_stride(primals_1410, (32, ), (1, ))
    assert_size_stride(primals_1411, (32, ), (1, ))
    assert_size_stride(primals_1412, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1413, (32, ), (1, ))
    assert_size_stride(primals_1414, (32, ), (1, ))
    assert_size_stride(primals_1415, (32, ), (1, ))
    assert_size_stride(primals_1416, (32, ), (1, ))
    assert_size_stride(primals_1417, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1418, (32, ), (1, ))
    assert_size_stride(primals_1419, (32, ), (1, ))
    assert_size_stride(primals_1420, (32, ), (1, ))
    assert_size_stride(primals_1421, (32, ), (1, ))
    assert_size_stride(primals_1422, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1423, (32, ), (1, ))
    assert_size_stride(primals_1424, (32, ), (1, ))
    assert_size_stride(primals_1425, (32, ), (1, ))
    assert_size_stride(primals_1426, (32, ), (1, ))
    assert_size_stride(primals_1427, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1428, (32, ), (1, ))
    assert_size_stride(primals_1429, (32, ), (1, ))
    assert_size_stride(primals_1430, (32, ), (1, ))
    assert_size_stride(primals_1431, (32, ), (1, ))
    assert_size_stride(primals_1432, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1433, (32, ), (1, ))
    assert_size_stride(primals_1434, (32, ), (1, ))
    assert_size_stride(primals_1435, (32, ), (1, ))
    assert_size_stride(primals_1436, (32, ), (1, ))
    assert_size_stride(primals_1437, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1438, (32, ), (1, ))
    assert_size_stride(primals_1439, (32, ), (1, ))
    assert_size_stride(primals_1440, (32, ), (1, ))
    assert_size_stride(primals_1441, (32, ), (1, ))
    assert_size_stride(primals_1442, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1443, (32, ), (1, ))
    assert_size_stride(primals_1444, (32, ), (1, ))
    assert_size_stride(primals_1445, (32, ), (1, ))
    assert_size_stride(primals_1446, (32, ), (1, ))
    assert_size_stride(primals_1447, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_1448, (4, ), (1, ))
    assert_size_stride(primals_1449, (4, ), (1, ))
    assert_size_stride(primals_1450, (4, ), (1, ))
    assert_size_stride(primals_1451, (4, ), (1, ))
    assert_size_stride(primals_1452, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_1453, (4, ), (1, ))
    assert_size_stride(primals_1454, (4, ), (1, ))
    assert_size_stride(primals_1455, (4, ), (1, ))
    assert_size_stride(primals_1456, (4, ), (1, ))
    assert_size_stride(primals_1457, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1458, (4, ), (1, ))
    assert_size_stride(primals_1459, (4, ), (1, ))
    assert_size_stride(primals_1460, (4, ), (1, ))
    assert_size_stride(primals_1461, (4, ), (1, ))
    assert_size_stride(primals_1462, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1463, (8, ), (1, ))
    assert_size_stride(primals_1464, (8, ), (1, ))
    assert_size_stride(primals_1465, (8, ), (1, ))
    assert_size_stride(primals_1466, (8, ), (1, ))
    assert_size_stride(primals_1467, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_1468, (8, ), (1, ))
    assert_size_stride(primals_1469, (8, ), (1, ))
    assert_size_stride(primals_1470, (8, ), (1, ))
    assert_size_stride(primals_1471, (8, ), (1, ))
    assert_size_stride(primals_1472, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1473, (8, ), (1, ))
    assert_size_stride(primals_1474, (8, ), (1, ))
    assert_size_stride(primals_1475, (8, ), (1, ))
    assert_size_stride(primals_1476, (8, ), (1, ))
    assert_size_stride(primals_1477, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1478, (4, ), (1, ))
    assert_size_stride(primals_1479, (4, ), (1, ))
    assert_size_stride(primals_1480, (4, ), (1, ))
    assert_size_stride(primals_1481, (4, ), (1, ))
    assert_size_stride(primals_1482, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1483, (16, ), (1, ))
    assert_size_stride(primals_1484, (16, ), (1, ))
    assert_size_stride(primals_1485, (16, ), (1, ))
    assert_size_stride(primals_1486, (16, ), (1, ))
    assert_size_stride(primals_1487, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1488, (16, ), (1, ))
    assert_size_stride(primals_1489, (16, ), (1, ))
    assert_size_stride(primals_1490, (16, ), (1, ))
    assert_size_stride(primals_1491, (16, ), (1, ))
    assert_size_stride(primals_1492, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1493, (16, ), (1, ))
    assert_size_stride(primals_1494, (16, ), (1, ))
    assert_size_stride(primals_1495, (16, ), (1, ))
    assert_size_stride(primals_1496, (16, ), (1, ))
    assert_size_stride(primals_1497, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1498, (4, ), (1, ))
    assert_size_stride(primals_1499, (4, ), (1, ))
    assert_size_stride(primals_1500, (4, ), (1, ))
    assert_size_stride(primals_1501, (4, ), (1, ))
    assert_size_stride(primals_1502, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1503, (4, ), (1, ))
    assert_size_stride(primals_1504, (4, ), (1, ))
    assert_size_stride(primals_1505, (4, ), (1, ))
    assert_size_stride(primals_1506, (4, ), (1, ))
    assert_size_stride(primals_1507, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_1508, (32, ), (1, ))
    assert_size_stride(primals_1509, (32, ), (1, ))
    assert_size_stride(primals_1510, (32, ), (1, ))
    assert_size_stride(primals_1511, (32, ), (1, ))
    assert_size_stride(primals_1512, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1513, (8, ), (1, ))
    assert_size_stride(primals_1514, (8, ), (1, ))
    assert_size_stride(primals_1515, (8, ), (1, ))
    assert_size_stride(primals_1516, (8, ), (1, ))
    assert_size_stride(primals_1517, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_1518, (32, ), (1, ))
    assert_size_stride(primals_1519, (32, ), (1, ))
    assert_size_stride(primals_1520, (32, ), (1, ))
    assert_size_stride(primals_1521, (32, ), (1, ))
    assert_size_stride(primals_1522, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_1523, (32, ), (1, ))
    assert_size_stride(primals_1524, (32, ), (1, ))
    assert_size_stride(primals_1525, (32, ), (1, ))
    assert_size_stride(primals_1526, (32, ), (1, ))
    assert_size_stride(primals_1527, (60, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_1528, (60, ), (1, ))
    assert_size_stride(primals_1529, (60, ), (1, ))
    assert_size_stride(primals_1530, (60, ), (1, ))
    assert_size_stride(primals_1531, (60, ), (1, ))
    assert_size_stride(primals_1532, (60, ), (1, ))
    assert_size_stride(primals_1533, (4, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_1534, (4, ), (1, ))
    assert_size_stride(primals_1535, (512, 60, 3, 3), (540, 9, 3, 1))
    assert_size_stride(primals_1536, (512, ), (1, ))
    assert_size_stride(primals_1537, (512, ), (1, ))
    assert_size_stride(primals_1538, (512, ), (1, ))
    assert_size_stride(primals_1539, (512, ), (1, ))
    assert_size_stride(primals_1540, (512, ), (1, ))
    assert_size_stride(primals_1541, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1542, (256, ), (1, ))
    assert_size_stride(primals_1543, (256, ), (1, ))
    assert_size_stride(primals_1544, (256, ), (1, ))
    assert_size_stride(primals_1545, (256, ), (1, ))
    assert_size_stride(primals_1546, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1547, (256, ), (1, ))
    assert_size_stride(primals_1548, (256, ), (1, ))
    assert_size_stride(primals_1549, (256, ), (1, ))
    assert_size_stride(primals_1550, (256, ), (1, ))
    assert_size_stride(primals_1551, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1552, (256, ), (1, ))
    assert_size_stride(primals_1553, (256, ), (1, ))
    assert_size_stride(primals_1554, (256, ), (1, ))
    assert_size_stride(primals_1555, (256, ), (1, ))
    assert_size_stride(primals_1556, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1557, (256, ), (1, ))
    assert_size_stride(primals_1558, (256, ), (1, ))
    assert_size_stride(primals_1559, (256, ), (1, ))
    assert_size_stride(primals_1560, (256, ), (1, ))
    assert_size_stride(primals_1561, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1562, (256, ), (1, ))
    assert_size_stride(primals_1563, (256, ), (1, ))
    assert_size_stride(primals_1564, (256, ), (1, ))
    assert_size_stride(primals_1565, (256, ), (1, ))
    assert_size_stride(primals_1566, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1567, (512, ), (1, ))
    assert_size_stride(primals_1568, (512, ), (1, ))
    assert_size_stride(primals_1569, (512, ), (1, ))
    assert_size_stride(primals_1570, (512, ), (1, ))
    assert_size_stride(primals_1571, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_1572, (512, ), (1, ))
    assert_size_stride(primals_1573, (512, ), (1, ))
    assert_size_stride(primals_1574, (512, ), (1, ))
    assert_size_stride(primals_1575, (512, ), (1, ))
    assert_size_stride(primals_1576, (4, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1577, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf2, primals_8, primals_9, primals_10, primals_11, buf3, 65536, grid=grid(65536), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf5 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf4, primals_13, primals_14, primals_15, primals_16, buf5, 65536, grid=grid(65536), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf7 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf6, primals_18, primals_19, primals_20, primals_21, buf7, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf3, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf10 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf11, buf8, primals_23, primals_24, primals_25, primals_26, buf9, primals_28, primals_29, primals_30, primals_31, 262144, grid=grid(262144), stream=stream0)
        del primals_26
        del primals_31
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf13 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf12, primals_33, primals_34, primals_35, primals_36, buf13, 65536, grid=grid(65536), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf15 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf14, primals_38, primals_39, primals_40, primals_41, buf15, 65536, grid=grid(65536), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf16, primals_43, primals_44, primals_45, primals_46, buf11, buf17, 262144, grid=grid(262144), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf18, primals_48, primals_49, primals_50, primals_51, buf19, 65536, grid=grid(65536), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf21 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf20, primals_53, primals_54, primals_55, primals_56, buf21, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf23 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf22, primals_58, primals_59, primals_60, primals_61, buf17, buf23, 262144, grid=grid(262144), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf25 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf24, primals_63, primals_64, primals_65, primals_66, buf25, 65536, grid=grid(65536), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf27 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf26, primals_68, primals_69, primals_70, primals_71, buf27, 65536, grid=grid(65536), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf29 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_37, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf28, primals_73, primals_74, primals_75, primals_76, buf23, buf29, 262144, grid=grid(262144), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf31 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf30, primals_78, primals_79, primals_80, primals_81, buf31, 4096, grid=grid(4096), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf29, primals_82, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 8, 8, 8), (512, 64, 8, 1))
        buf33 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf32, primals_83, primals_84, primals_85, primals_86, buf33, 2048, grid=grid(2048), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf31, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf35 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf34, primals_88, primals_89, primals_90, primals_91, buf35, 4096, grid=grid(4096), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf37 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf36, primals_93, primals_94, primals_95, primals_96, buf31, buf37, 4096, grid=grid(4096), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf39 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_48, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf38, primals_98, primals_99, primals_100, primals_101, buf39, 4096, grid=grid(4096), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf41 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf40, primals_103, primals_104, primals_105, primals_106, buf37, buf41, 4096, grid=grid(4096), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf43 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf42, primals_108, primals_109, primals_110, primals_111, buf43, 4096, grid=grid(4096), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf45 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_58, out_59, out_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf44, primals_113, primals_114, primals_115, primals_116, buf41, buf45, 4096, grid=grid(4096), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf47 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf46, primals_118, primals_119, primals_120, primals_121, buf47, 4096, grid=grid(4096), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf33, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 8, 8, 8), (512, 64, 8, 1))
        buf51 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_69, out_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf50, primals_128, primals_129, primals_130, primals_131, buf51, 2048, grid=grid(2048), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 8, 8, 8), (512, 64, 8, 1))
        buf53 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_72, out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf52, primals_133, primals_134, primals_135, primals_136, buf33, buf53, 2048, grid=grid(2048), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 8, 8, 8), (512, 64, 8, 1))
        buf55 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_76, out_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf54, primals_138, primals_139, primals_140, primals_141, buf55, 2048, grid=grid(2048), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [out_78], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 8, 8, 8), (512, 64, 8, 1))
        buf57 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_79, out_80, out_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf56, primals_143, primals_144, primals_145, primals_146, buf53, buf57, 2048, grid=grid(2048), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_82], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 8, 8, 8), (512, 64, 8, 1))
        buf59 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_83, out_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf58, primals_148, primals_149, primals_150, primals_151, buf59, 2048, grid=grid(2048), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 8, 8, 8), (512, 64, 8, 1))
        buf61 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_86, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf60, primals_153, primals_154, primals_155, primals_156, buf57, buf61, 2048, grid=grid(2048), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [out_89], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 8, 8, 8), (512, 64, 8, 1))
        buf63 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_90, out_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf62, primals_158, primals_159, primals_160, primals_161, buf63, 2048, grid=grid(2048), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 8, 8, 8), (512, 64, 8, 1))
        buf65 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_93, out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf64, primals_163, primals_164, primals_165, primals_166, buf61, buf65, 2048, grid=grid(2048), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 4, 8, 8), (256, 64, 8, 1))
        buf67 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf66, primals_168, primals_169, primals_170, primals_171, buf67, 1024, grid=grid(1024), stream=stream0)
        del primals_171
        buf68 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf68, 16, grid=grid(16), stream=stream0)
        buf69 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_10.run(buf69, 16, grid=grid(16), stream=stream0)
        buf70 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf70, 16, grid=grid(16), stream=stream0)
        buf71 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_10.run(buf71, 16, grid=grid(16), stream=stream0)
        buf72 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_11.run(buf72, 16, grid=grid(16), stream=stream0)
        buf74 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_11.run(buf74, 16, grid=grid(16), stream=stream0)
        buf49 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf73 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf75 = buf73; del buf73  # reuse
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [out_65, out_66, out_67, interpolate, y, residual], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_12.run(buf76, buf48, primals_123, primals_124, primals_125, primals_126, buf45, buf68, buf70, buf67, buf71, buf72, buf69, buf74, buf49, 4096, grid=grid(4096), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf49, primals_172, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 8, 8, 8), (512, 64, 8, 1))
        buf78 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, y_1, residual_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf77, primals_173, primals_174, primals_175, primals_176, buf65, buf78, 2048, grid=grid(2048), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_177, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 16, 4, 4), (256, 16, 4, 1))
        buf80 = reinterpret_tensor(buf67, (4, 16, 4, 4), (256, 16, 4, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf79, primals_178, primals_179, primals_180, primals_181, buf80, 1024, grid=grid(1024), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf76, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf82 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf81, primals_183, primals_184, primals_185, primals_186, buf82, 4096, grid=grid(4096), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [out_99], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf84 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_100, out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf83, primals_188, primals_189, primals_190, primals_191, buf76, buf84, 4096, grid=grid(4096), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf86 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf85, primals_193, primals_194, primals_195, primals_196, buf86, 4096, grid=grid(4096), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf88 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf87, primals_198, primals_199, primals_200, primals_201, buf84, buf88, 4096, grid=grid(4096), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf90 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf89, primals_203, primals_204, primals_205, primals_206, buf90, 4096, grid=grid(4096), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf92 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf91, primals_208, primals_209, primals_210, primals_211, buf88, buf92, 4096, grid=grid(4096), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf94 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf93, primals_213, primals_214, primals_215, primals_216, buf94, 4096, grid=grid(4096), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf78, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 8, 8, 8), (512, 64, 8, 1))
        buf98 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf97, primals_223, primals_224, primals_225, primals_226, buf98, 2048, grid=grid(2048), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [out_127], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 8, 8, 8), (512, 64, 8, 1))
        buf100 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_128, out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf99, primals_228, primals_229, primals_230, primals_231, buf78, buf100, 2048, grid=grid(2048), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 8, 8, 8), (512, 64, 8, 1))
        buf102 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf101, primals_233, primals_234, primals_235, primals_236, buf102, 2048, grid=grid(2048), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 8, 8, 8), (512, 64, 8, 1))
        buf104 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_135, out_136, out_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf103, primals_238, primals_239, primals_240, primals_241, buf100, buf104, 2048, grid=grid(2048), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [out_138], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 8, 8, 8), (512, 64, 8, 1))
        buf106 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_139, out_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf105, primals_243, primals_244, primals_245, primals_246, buf106, 2048, grid=grid(2048), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 8, 8, 8), (512, 64, 8, 1))
        buf108 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_142, out_143, out_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf107, primals_248, primals_249, primals_250, primals_251, buf104, buf108, 2048, grid=grid(2048), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [out_145], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 8, 8, 8), (512, 64, 8, 1))
        buf110 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_146, out_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf109, primals_253, primals_254, primals_255, primals_256, buf110, 2048, grid=grid(2048), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 8, 8, 8), (512, 64, 8, 1))
        buf112 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_149, out_150, out_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf111, primals_258, primals_259, primals_260, primals_261, buf108, buf112, 2048, grid=grid(2048), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf112, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 4, 8, 8), (256, 64, 8, 1))
        buf130 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf129, primals_303, primals_304, primals_305, primals_306, buf130, 1024, grid=grid(1024), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf80, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 16, 4, 4), (256, 16, 4, 1))
        buf114 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf113, primals_263, primals_264, primals_265, primals_266, buf114, 1024, grid=grid(1024), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [out_155], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 16, 4, 4), (256, 16, 4, 1))
        buf116 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_156, out_157, out_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf115, primals_268, primals_269, primals_270, primals_271, buf80, buf116, 1024, grid=grid(1024), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [out_159], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 16, 4, 4), (256, 16, 4, 1))
        buf118 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_160, out_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf117, primals_273, primals_274, primals_275, primals_276, buf118, 1024, grid=grid(1024), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [out_162], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 16, 4, 4), (256, 16, 4, 1))
        buf120 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, out_164, out_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf119, primals_278, primals_279, primals_280, primals_281, buf116, buf120, 1024, grid=grid(1024), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 16, 4, 4), (256, 16, 4, 1))
        buf122 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_167, out_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf121, primals_283, primals_284, primals_285, primals_286, buf122, 1024, grid=grid(1024), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [out_169], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 16, 4, 4), (256, 16, 4, 1))
        buf124 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_170, out_171, out_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf123, primals_288, primals_289, primals_290, primals_291, buf120, buf124, 1024, grid=grid(1024), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [out_173], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 16, 4, 4), (256, 16, 4, 1))
        buf126 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf125, primals_293, primals_294, primals_295, primals_296, buf126, 1024, grid=grid(1024), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 16, 4, 4), (256, 16, 4, 1))
        buf128 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_177, out_178, out_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf127, primals_298, primals_299, primals_300, primals_301, buf124, buf128, 1024, grid=grid(1024), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf128, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 4, 4, 4), (64, 16, 4, 1))
        buf134 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf133, primals_308, primals_309, primals_310, primals_311, buf134, 256, grid=grid(256), stream=stream0)
        del primals_311
        buf135 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf135, 16, grid=grid(16), stream=stream0)
        buf136 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_17.run(buf136, 16, grid=grid(16), stream=stream0)
        buf137 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate, interpolate_2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf137, 16, grid=grid(16), stream=stream0)
        buf138 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_17.run(buf138, 16, grid=grid(16), stream=stream0)
        buf139 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate, interpolate_2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_18.run(buf139, 16, grid=grid(16), stream=stream0)
        buf141 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_2], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_18.run(buf141, 16, grid=grid(16), stream=stream0)
        buf96 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf131 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf132 = buf131; del buf131  # reuse
        buf142 = buf132; del buf132  # reuse
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [out_121, out_122, out_123, interpolate_1, y_2, interpolate_2, y_3, residual_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19.run(buf143, buf95, primals_218, primals_219, primals_220, primals_221, buf92, buf68, buf70, buf130, buf71, buf72, buf69, buf74, buf135, buf137, buf134, buf138, buf139, buf136, buf141, buf96, 4096, grid=grid(4096), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf96, primals_312, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf128, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 8, 4, 4), (128, 16, 4, 1))
        buf146 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf145, primals_318, primals_319, primals_320, primals_321, buf146, 512, grid=grid(512), stream=stream0)
        del primals_321
        buf147 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(buf147, 8, grid=grid(8), stream=stream0)
        buf148 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf148, 8, grid=grid(8), stream=stream0)
        buf149 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(buf149, 8, grid=grid(8), stream=stream0)
        buf150 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf150, 8, grid=grid(8), stream=stream0)
        buf151 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_23.run(buf151, 8, grid=grid(8), stream=stream0)
        buf153 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_23.run(buf153, 8, grid=grid(8), stream=stream0)
        buf152 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf154 = buf152; del buf152  # reuse
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_21, y_4, interpolate_3, y_5, residual_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24.run(buf155, buf147, buf149, buf146, buf150, buf151, buf144, primals_313, primals_314, primals_315, primals_316, buf112, buf148, buf153, 2048, grid=grid(2048), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf96, primals_322, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 4, 8, 8), (256, 64, 8, 1))
        buf157 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf156, primals_323, primals_324, primals_325, primals_326, buf157, 1024, grid=grid(1024), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_327, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf112, primals_332, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 16, 4, 4), (256, 16, 4, 1))
        buf160 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_30, y_6, y_7, residual_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf161, buf158, primals_328, primals_329, primals_330, primals_331, buf159, primals_333, primals_334, primals_335, primals_336, buf128, 1024, grid=grid(1024), stream=stream0)
        del primals_331
        del primals_336
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf143, primals_337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf163 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_181, out_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf162, primals_338, primals_339, primals_340, primals_341, buf163, 4096, grid=grid(4096), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [out_183], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf165 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_184, out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf164, primals_343, primals_344, primals_345, primals_346, buf143, buf165, 4096, grid=grid(4096), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [out_187], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf167 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_188, out_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf166, primals_348, primals_349, primals_350, primals_351, buf167, 4096, grid=grid(4096), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf169 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_191, out_192, out_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf168, primals_353, primals_354, primals_355, primals_356, buf165, buf169, 4096, grid=grid(4096), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [out_194], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf171 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_195, out_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf170, primals_358, primals_359, primals_360, primals_361, buf171, 4096, grid=grid(4096), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [out_197], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf173 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_198, out_199, out_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf172, primals_363, primals_364, primals_365, primals_366, buf169, buf173, 4096, grid=grid(4096), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [out_201], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf175 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_202, out_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf174, primals_368, primals_369, primals_370, primals_371, buf175, 4096, grid=grid(4096), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf155, primals_377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 8, 8, 8), (512, 64, 8, 1))
        buf179 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf178, primals_378, primals_379, primals_380, primals_381, buf179, 2048, grid=grid(2048), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [out_211], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 8, 8, 8), (512, 64, 8, 1))
        buf181 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_212, out_213, out_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf180, primals_383, primals_384, primals_385, primals_386, buf155, buf181, 2048, grid=grid(2048), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [out_215], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 8, 8, 8), (512, 64, 8, 1))
        buf183 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_216, out_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf182, primals_388, primals_389, primals_390, primals_391, buf183, 2048, grid=grid(2048), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [out_218], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 8, 8, 8), (512, 64, 8, 1))
        buf185 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_219, out_220, out_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf184, primals_393, primals_394, primals_395, primals_396, buf181, buf185, 2048, grid=grid(2048), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [out_222], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 8, 8, 8), (512, 64, 8, 1))
        buf187 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_223, out_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf186, primals_398, primals_399, primals_400, primals_401, buf187, 2048, grid=grid(2048), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [out_225], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 8, 8, 8), (512, 64, 8, 1))
        buf189 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_226, out_227, out_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf188, primals_403, primals_404, primals_405, primals_406, buf185, buf189, 2048, grid=grid(2048), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [out_229], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 8, 8, 8), (512, 64, 8, 1))
        buf191 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_230, out_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf190, primals_408, primals_409, primals_410, primals_411, buf191, 2048, grid=grid(2048), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 8, 8, 8), (512, 64, 8, 1))
        buf193 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_233, out_234, out_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf192, primals_413, primals_414, primals_415, primals_416, buf189, buf193, 2048, grid=grid(2048), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf193, primals_457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 4, 8, 8), (256, 64, 8, 1))
        buf211 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf210, primals_458, primals_459, primals_460, primals_461, buf211, 1024, grid=grid(1024), stream=stream0)
        del primals_461
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf161, primals_417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 16, 4, 4), (256, 16, 4, 1))
        buf195 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_237, out_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf194, primals_418, primals_419, primals_420, primals_421, buf195, 1024, grid=grid(1024), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [out_239], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 16, 4, 4), (256, 16, 4, 1))
        buf197 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_240, out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf196, primals_423, primals_424, primals_425, primals_426, buf161, buf197, 1024, grid=grid(1024), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 16, 4, 4), (256, 16, 4, 1))
        buf199 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_244, out_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf198, primals_428, primals_429, primals_430, primals_431, buf199, 1024, grid=grid(1024), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [out_246], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 16, 4, 4), (256, 16, 4, 1))
        buf201 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_247, out_248, out_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf200, primals_433, primals_434, primals_435, primals_436, buf197, buf201, 1024, grid=grid(1024), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_437, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 16, 4, 4), (256, 16, 4, 1))
        buf203 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_251, out_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf202, primals_438, primals_439, primals_440, primals_441, buf203, 1024, grid=grid(1024), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [out_253], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 16, 4, 4), (256, 16, 4, 1))
        buf205 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_254, out_255, out_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf204, primals_443, primals_444, primals_445, primals_446, buf201, buf205, 1024, grid=grid(1024), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [out_257], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_447, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 16, 4, 4), (256, 16, 4, 1))
        buf207 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_258, out_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf206, primals_448, primals_449, primals_450, primals_451, buf207, 1024, grid=grid(1024), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 16, 4, 4), (256, 16, 4, 1))
        buf209 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_261, out_262, out_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf208, primals_453, primals_454, primals_455, primals_456, buf205, buf209, 1024, grid=grid(1024), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf209, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 4, 4, 4), (64, 16, 4, 1))
        buf215 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf214, primals_463, primals_464, primals_465, primals_466, buf215, 256, grid=grid(256), stream=stream0)
        del primals_466
        buf177 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf212 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf213 = buf212; del buf212  # reuse
        buf217 = buf213; del buf213  # reuse
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [out_205, out_206, out_207, interpolate_4, y_8, interpolate_5, y_9, residual_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19.run(buf218, buf176, primals_373, primals_374, primals_375, primals_376, buf173, buf68, buf70, buf211, buf71, buf72, buf69, buf74, buf135, buf137, buf215, buf138, buf139, buf136, buf141, buf177, 4096, grid=grid(4096), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf177, primals_467, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf209, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 8, 4, 4), (128, 16, 4, 1))
        buf221 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf220, primals_473, primals_474, primals_475, primals_476, buf221, 512, grid=grid(512), stream=stream0)
        del primals_476
        buf222 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf223 = buf222; del buf222  # reuse
        buf224 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [input_36, y_10, interpolate_6, y_11, residual_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24.run(buf224, buf147, buf149, buf221, buf150, buf151, buf219, primals_468, primals_469, primals_470, primals_471, buf193, buf148, buf153, 2048, grid=grid(2048), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf177, primals_477, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 4, 8, 8), (256, 64, 8, 1))
        buf226 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf225, primals_478, primals_479, primals_480, primals_481, buf226, 1024, grid=grid(1024), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_482, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf193, primals_487, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 16, 4, 4), (256, 16, 4, 1))
        buf229 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf230 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_45, y_12, y_13, residual_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf230, buf227, primals_483, primals_484, primals_485, primals_486, buf228, primals_488, primals_489, primals_490, primals_491, buf209, 1024, grid=grid(1024), stream=stream0)
        del primals_486
        del primals_491
        # Topologically Sorted Source Nodes: [out_264], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf218, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf232 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf231, primals_493, primals_494, primals_495, primals_496, buf232, 4096, grid=grid(4096), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [out_267], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf234 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_268, out_269, out_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf233, primals_498, primals_499, primals_500, primals_501, buf218, buf234, 4096, grid=grid(4096), stream=stream0)
        del primals_501
        # Topologically Sorted Source Nodes: [out_271], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_502, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf236 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_272, out_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf235, primals_503, primals_504, primals_505, primals_506, buf236, 4096, grid=grid(4096), stream=stream0)
        del primals_506
        # Topologically Sorted Source Nodes: [out_274], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_507, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf238 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_275, out_276, out_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf237, primals_508, primals_509, primals_510, primals_511, buf234, buf238, 4096, grid=grid(4096), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [out_278], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf240 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_279, out_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf239, primals_513, primals_514, primals_515, primals_516, buf240, 4096, grid=grid(4096), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [out_281], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_517, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf242 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_282, out_283, out_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf241, primals_518, primals_519, primals_520, primals_521, buf238, buf242, 4096, grid=grid(4096), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [out_285], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf244 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_286, out_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf243, primals_523, primals_524, primals_525, primals_526, buf244, 4096, grid=grid(4096), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [out_288], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_527, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_292], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf224, primals_532, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 8, 8, 8), (512, 64, 8, 1))
        buf248 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_293, out_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf247, primals_533, primals_534, primals_535, primals_536, buf248, 2048, grid=grid(2048), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [out_295], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_537, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 8, 8, 8), (512, 64, 8, 1))
        buf250 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_296, out_297, out_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf249, primals_538, primals_539, primals_540, primals_541, buf224, buf250, 2048, grid=grid(2048), stream=stream0)
        del primals_541
        # Topologically Sorted Source Nodes: [out_299], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_542, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 8, 8, 8), (512, 64, 8, 1))
        buf252 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_300, out_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf251, primals_543, primals_544, primals_545, primals_546, buf252, 2048, grid=grid(2048), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [out_302], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_547, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 8, 8, 8), (512, 64, 8, 1))
        buf254 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_303, out_304, out_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf253, primals_548, primals_549, primals_550, primals_551, buf250, buf254, 2048, grid=grid(2048), stream=stream0)
        del primals_551
        # Topologically Sorted Source Nodes: [out_306], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 8, 8, 8), (512, 64, 8, 1))
        buf256 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_307, out_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf255, primals_553, primals_554, primals_555, primals_556, buf256, 2048, grid=grid(2048), stream=stream0)
        del primals_556
        # Topologically Sorted Source Nodes: [out_309], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_557, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 8, 8, 8), (512, 64, 8, 1))
        buf258 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_310, out_311, out_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf257, primals_558, primals_559, primals_560, primals_561, buf254, buf258, 2048, grid=grid(2048), stream=stream0)
        del primals_561
        # Topologically Sorted Source Nodes: [out_313], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_562, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 8, 8, 8), (512, 64, 8, 1))
        buf260 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_314, out_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf259, primals_563, primals_564, primals_565, primals_566, buf260, 2048, grid=grid(2048), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [out_316], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_567, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 8, 8, 8), (512, 64, 8, 1))
        buf262 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_317, out_318, out_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf261, primals_568, primals_569, primals_570, primals_571, buf258, buf262, 2048, grid=grid(2048), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf262, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 4, 8, 8), (256, 64, 8, 1))
        buf280 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf279, primals_613, primals_614, primals_615, primals_616, buf280, 1024, grid=grid(1024), stream=stream0)
        del primals_616
        # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf230, primals_572, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 16, 4, 4), (256, 16, 4, 1))
        buf264 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf263, primals_573, primals_574, primals_575, primals_576, buf264, 1024, grid=grid(1024), stream=stream0)
        del primals_576
        # Topologically Sorted Source Nodes: [out_323], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_577, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 16, 4, 4), (256, 16, 4, 1))
        buf266 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf265, primals_578, primals_579, primals_580, primals_581, buf230, buf266, 1024, grid=grid(1024), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [out_327], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_582, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 16, 4, 4), (256, 16, 4, 1))
        buf268 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_328, out_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf267, primals_583, primals_584, primals_585, primals_586, buf268, 1024, grid=grid(1024), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [out_330], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_587, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 16, 4, 4), (256, 16, 4, 1))
        buf270 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_331, out_332, out_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf269, primals_588, primals_589, primals_590, primals_591, buf266, buf270, 1024, grid=grid(1024), stream=stream0)
        del primals_591
        # Topologically Sorted Source Nodes: [out_334], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_592, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 16, 4, 4), (256, 16, 4, 1))
        buf272 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_335, out_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf271, primals_593, primals_594, primals_595, primals_596, buf272, 1024, grid=grid(1024), stream=stream0)
        del primals_596
        # Topologically Sorted Source Nodes: [out_337], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_597, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 16, 4, 4), (256, 16, 4, 1))
        buf274 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_338, out_339, out_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf273, primals_598, primals_599, primals_600, primals_601, buf270, buf274, 1024, grid=grid(1024), stream=stream0)
        del primals_601
        # Topologically Sorted Source Nodes: [out_341], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_602, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 16, 4, 4), (256, 16, 4, 1))
        buf276 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_342, out_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf275, primals_603, primals_604, primals_605, primals_606, buf276, 1024, grid=grid(1024), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [out_344], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_607, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 16, 4, 4), (256, 16, 4, 1))
        buf278 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_345, out_346, out_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf277, primals_608, primals_609, primals_610, primals_611, buf274, buf278, 1024, grid=grid(1024), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf278, primals_617, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 4, 4, 4), (64, 16, 4, 1))
        buf284 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf283, primals_618, primals_619, primals_620, primals_621, buf284, 256, grid=grid(256), stream=stream0)
        del primals_621
        buf246 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf281 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf282 = buf281; del buf281  # reuse
        buf286 = buf282; del buf282  # reuse
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [out_289, out_290, out_291, interpolate_7, y_14, interpolate_8, y_15, residual_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19.run(buf287, buf245, primals_528, primals_529, primals_530, primals_531, buf242, buf68, buf70, buf280, buf71, buf72, buf69, buf74, buf135, buf137, buf284, buf138, buf139, buf136, buf141, buf246, 4096, grid=grid(4096), stream=stream0)
        del primals_531
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf246, primals_622, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf278, primals_627, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 8, 4, 4), (128, 16, 4, 1))
        buf290 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf289, primals_628, primals_629, primals_630, primals_631, buf290, 512, grid=grid(512), stream=stream0)
        del primals_631
        buf291 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf292 = buf291; del buf291  # reuse
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [input_51, y_16, interpolate_9, y_17, residual_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24.run(buf293, buf147, buf149, buf290, buf150, buf151, buf288, primals_623, primals_624, primals_625, primals_626, buf262, buf148, buf153, 2048, grid=grid(2048), stream=stream0)
        del primals_626
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf246, primals_632, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 4, 8, 8), (256, 64, 8, 1))
        buf295 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf294, primals_633, primals_634, primals_635, primals_636, buf295, 1024, grid=grid(1024), stream=stream0)
        del primals_636
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_637, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf262, primals_642, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 16, 4, 4), (256, 16, 4, 1))
        buf298 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [input_58, input_60, y_18, y_19, residual_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf299, buf296, primals_638, primals_639, primals_640, primals_641, buf297, primals_643, primals_644, primals_645, primals_646, buf278, 1024, grid=grid(1024), stream=stream0)
        del primals_641
        del primals_646
        # Topologically Sorted Source Nodes: [out_348], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf287, primals_647, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf301 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_349, out_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf300, primals_648, primals_649, primals_650, primals_651, buf301, 4096, grid=grid(4096), stream=stream0)
        del primals_651
        # Topologically Sorted Source Nodes: [out_351], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_652, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf303 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_352, out_353, out_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf302, primals_653, primals_654, primals_655, primals_656, buf287, buf303, 4096, grid=grid(4096), stream=stream0)
        del primals_656
        # Topologically Sorted Source Nodes: [out_355], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_657, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf305 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_356, out_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf304, primals_658, primals_659, primals_660, primals_661, buf305, 4096, grid=grid(4096), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [out_358], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_662, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf307 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_359, out_360, out_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf306, primals_663, primals_664, primals_665, primals_666, buf303, buf307, 4096, grid=grid(4096), stream=stream0)
        del primals_666
        # Topologically Sorted Source Nodes: [out_362], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf309 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_363, out_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf308, primals_668, primals_669, primals_670, primals_671, buf309, 4096, grid=grid(4096), stream=stream0)
        del primals_671
        # Topologically Sorted Source Nodes: [out_365], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_672, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf311 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_366, out_367, out_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf310, primals_673, primals_674, primals_675, primals_676, buf307, buf311, 4096, grid=grid(4096), stream=stream0)
        del primals_676
        # Topologically Sorted Source Nodes: [out_369], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_677, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf313 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_370, out_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf312, primals_678, primals_679, primals_680, primals_681, buf313, 4096, grid=grid(4096), stream=stream0)
        del primals_681
        # Topologically Sorted Source Nodes: [out_372], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_682, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_376], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf293, primals_687, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 8, 8, 8), (512, 64, 8, 1))
        buf317 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_377, out_378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf316, primals_688, primals_689, primals_690, primals_691, buf317, 2048, grid=grid(2048), stream=stream0)
        del primals_691
        # Topologically Sorted Source Nodes: [out_379], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_692, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 8, 8, 8), (512, 64, 8, 1))
        buf319 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_380, out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf318, primals_693, primals_694, primals_695, primals_696, buf293, buf319, 2048, grid=grid(2048), stream=stream0)
        del primals_696
        # Topologically Sorted Source Nodes: [out_383], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_697, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 8, 8, 8), (512, 64, 8, 1))
        buf321 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_384, out_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf320, primals_698, primals_699, primals_700, primals_701, buf321, 2048, grid=grid(2048), stream=stream0)
        del primals_701
        # Topologically Sorted Source Nodes: [out_386], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_702, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 8, 8, 8), (512, 64, 8, 1))
        buf323 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_387, out_388, out_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf322, primals_703, primals_704, primals_705, primals_706, buf319, buf323, 2048, grid=grid(2048), stream=stream0)
        del primals_706
        # Topologically Sorted Source Nodes: [out_390], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_707, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 8, 8, 8), (512, 64, 8, 1))
        buf325 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_391, out_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf324, primals_708, primals_709, primals_710, primals_711, buf325, 2048, grid=grid(2048), stream=stream0)
        del primals_711
        # Topologically Sorted Source Nodes: [out_393], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_712, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 8, 8, 8), (512, 64, 8, 1))
        buf327 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_394, out_395, out_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf326, primals_713, primals_714, primals_715, primals_716, buf323, buf327, 2048, grid=grid(2048), stream=stream0)
        del primals_716
        # Topologically Sorted Source Nodes: [out_397], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_717, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 8, 8, 8), (512, 64, 8, 1))
        buf329 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_398, out_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf328, primals_718, primals_719, primals_720, primals_721, buf329, 2048, grid=grid(2048), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [out_400], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_722, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 8, 8, 8), (512, 64, 8, 1))
        buf331 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_401, out_402, out_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf330, primals_723, primals_724, primals_725, primals_726, buf327, buf331, 2048, grid=grid(2048), stream=stream0)
        del primals_726
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf331, primals_767, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 4, 8, 8), (256, 64, 8, 1))
        buf349 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf348, primals_768, primals_769, primals_770, primals_771, buf349, 1024, grid=grid(1024), stream=stream0)
        del primals_771
        # Topologically Sorted Source Nodes: [out_404], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf299, primals_727, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 16, 4, 4), (256, 16, 4, 1))
        buf333 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_405, out_406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf332, primals_728, primals_729, primals_730, primals_731, buf333, 1024, grid=grid(1024), stream=stream0)
        del primals_731
        # Topologically Sorted Source Nodes: [out_407], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_732, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 16, 4, 4), (256, 16, 4, 1))
        buf335 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_408, out_409, out_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf334, primals_733, primals_734, primals_735, primals_736, buf299, buf335, 1024, grid=grid(1024), stream=stream0)
        del primals_736
        # Topologically Sorted Source Nodes: [out_411], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_737, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 16, 4, 4), (256, 16, 4, 1))
        buf337 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_412, out_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf336, primals_738, primals_739, primals_740, primals_741, buf337, 1024, grid=grid(1024), stream=stream0)
        del primals_741
        # Topologically Sorted Source Nodes: [out_414], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_742, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 16, 4, 4), (256, 16, 4, 1))
        buf339 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_415, out_416, out_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf338, primals_743, primals_744, primals_745, primals_746, buf335, buf339, 1024, grid=grid(1024), stream=stream0)
        del primals_746
        # Topologically Sorted Source Nodes: [out_418], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_747, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 16, 4, 4), (256, 16, 4, 1))
        buf341 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_419, out_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf340, primals_748, primals_749, primals_750, primals_751, buf341, 1024, grid=grid(1024), stream=stream0)
        del primals_751
        # Topologically Sorted Source Nodes: [out_421], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_752, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 16, 4, 4), (256, 16, 4, 1))
        buf343 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_422, out_423, out_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf342, primals_753, primals_754, primals_755, primals_756, buf339, buf343, 1024, grid=grid(1024), stream=stream0)
        del primals_756
        # Topologically Sorted Source Nodes: [out_425], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_757, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 16, 4, 4), (256, 16, 4, 1))
        buf345 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_426, out_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf344, primals_758, primals_759, primals_760, primals_761, buf345, 1024, grid=grid(1024), stream=stream0)
        del primals_761
        # Topologically Sorted Source Nodes: [out_428], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_762, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 16, 4, 4), (256, 16, 4, 1))
        buf347 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_429, out_430, out_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf346, primals_763, primals_764, primals_765, primals_766, buf343, buf347, 1024, grid=grid(1024), stream=stream0)
        del primals_766
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf347, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 4, 4, 4), (64, 16, 4, 1))
        buf353 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf352, primals_773, primals_774, primals_775, primals_776, buf353, 256, grid=grid(256), stream=stream0)
        del primals_776
        buf315 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf350 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf351 = buf350; del buf350  # reuse
        buf355 = buf351; del buf351  # reuse
        buf356 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [out_373, out_374, out_375, interpolate_10, y_20, interpolate_11, y_21, residual_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19.run(buf356, buf314, primals_683, primals_684, primals_685, primals_686, buf311, buf68, buf70, buf349, buf71, buf72, buf69, buf74, buf135, buf137, buf353, buf138, buf139, buf136, buf141, buf315, 4096, grid=grid(4096), stream=stream0)
        del primals_686
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf315, primals_777, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf347, primals_782, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 8, 4, 4), (128, 16, 4, 1))
        buf359 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf358, primals_783, primals_784, primals_785, primals_786, buf359, 512, grid=grid(512), stream=stream0)
        del primals_786
        buf360 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf361 = buf360; del buf360  # reuse
        buf362 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_66, y_22, interpolate_12, y_23, residual_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_24.run(buf362, buf147, buf149, buf359, buf150, buf151, buf357, primals_778, primals_779, primals_780, primals_781, buf331, buf148, buf153, 2048, grid=grid(2048), stream=stream0)
        del primals_781
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf315, primals_787, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 4, 8, 8), (256, 64, 8, 1))
        buf364 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [input_70, input_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf363, primals_788, primals_789, primals_790, primals_791, buf364, 1024, grid=grid(1024), stream=stream0)
        del primals_791
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_792, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf331, primals_797, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 16, 4, 4), (256, 16, 4, 1))
        buf367 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf368 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_75, y_24, y_25, residual_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf368, buf365, primals_793, primals_794, primals_795, primals_796, buf366, primals_798, primals_799, primals_800, primals_801, buf347, 1024, grid=grid(1024), stream=stream0)
        del primals_796
        del primals_801
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_802, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 32, 2, 2), (128, 4, 2, 1))
        buf370 = reinterpret_tensor(buf359, (4, 32, 2, 2), (128, 4, 2, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf369, primals_803, primals_804, primals_805, primals_806, buf370, 512, grid=grid(512), stream=stream0)
        del primals_806
        # Topologically Sorted Source Nodes: [out_432], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf356, primals_807, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf372 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_433, out_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf371, primals_808, primals_809, primals_810, primals_811, buf372, 4096, grid=grid(4096), stream=stream0)
        del primals_811
        # Topologically Sorted Source Nodes: [out_435], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, primals_812, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf374 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_436, out_437, out_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf373, primals_813, primals_814, primals_815, primals_816, buf356, buf374, 4096, grid=grid(4096), stream=stream0)
        del primals_816
        # Topologically Sorted Source Nodes: [out_439], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_817, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf376 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_440, out_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf375, primals_818, primals_819, primals_820, primals_821, buf376, 4096, grid=grid(4096), stream=stream0)
        del primals_821
        # Topologically Sorted Source Nodes: [out_442], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_822, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf378 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_443, out_444, out_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf377, primals_823, primals_824, primals_825, primals_826, buf374, buf378, 4096, grid=grid(4096), stream=stream0)
        del primals_826
        # Topologically Sorted Source Nodes: [out_446], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_827, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf380 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_447, out_448], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf379, primals_828, primals_829, primals_830, primals_831, buf380, 4096, grid=grid(4096), stream=stream0)
        del primals_831
        # Topologically Sorted Source Nodes: [out_449], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_832, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf382 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_450, out_451, out_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf381, primals_833, primals_834, primals_835, primals_836, buf378, buf382, 4096, grid=grid(4096), stream=stream0)
        del primals_836
        # Topologically Sorted Source Nodes: [out_453], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_837, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf384 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_454, out_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf383, primals_838, primals_839, primals_840, primals_841, buf384, 4096, grid=grid(4096), stream=stream0)
        del primals_841
        # Topologically Sorted Source Nodes: [out_456], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_842, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf362, primals_847, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 8, 8, 8), (512, 64, 8, 1))
        buf388 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_461, out_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf387, primals_848, primals_849, primals_850, primals_851, buf388, 2048, grid=grid(2048), stream=stream0)
        del primals_851
        # Topologically Sorted Source Nodes: [out_463], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_852, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 8, 8, 8), (512, 64, 8, 1))
        buf390 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_464, out_465, out_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf389, primals_853, primals_854, primals_855, primals_856, buf362, buf390, 2048, grid=grid(2048), stream=stream0)
        del primals_856
        # Topologically Sorted Source Nodes: [out_467], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, primals_857, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (4, 8, 8, 8), (512, 64, 8, 1))
        buf392 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_468, out_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf391, primals_858, primals_859, primals_860, primals_861, buf392, 2048, grid=grid(2048), stream=stream0)
        del primals_861
        # Topologically Sorted Source Nodes: [out_470], Original ATen: [aten.convolution]
        buf393 = extern_kernels.convolution(buf392, primals_862, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (4, 8, 8, 8), (512, 64, 8, 1))
        buf394 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_471, out_472, out_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf393, primals_863, primals_864, primals_865, primals_866, buf390, buf394, 2048, grid=grid(2048), stream=stream0)
        del primals_866
        # Topologically Sorted Source Nodes: [out_474], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_867, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 8, 8, 8), (512, 64, 8, 1))
        buf396 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_475, out_476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf395, primals_868, primals_869, primals_870, primals_871, buf396, 2048, grid=grid(2048), stream=stream0)
        del primals_871
        # Topologically Sorted Source Nodes: [out_477], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_872, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 8, 8, 8), (512, 64, 8, 1))
        buf398 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_478, out_479, out_480], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf397, primals_873, primals_874, primals_875, primals_876, buf394, buf398, 2048, grid=grid(2048), stream=stream0)
        del primals_876
        # Topologically Sorted Source Nodes: [out_481], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_877, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (4, 8, 8, 8), (512, 64, 8, 1))
        buf400 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_482, out_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf399, primals_878, primals_879, primals_880, primals_881, buf400, 2048, grid=grid(2048), stream=stream0)
        del primals_881
        # Topologically Sorted Source Nodes: [out_484], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_882, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 8, 8, 8), (512, 64, 8, 1))
        buf402 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_485, out_486, out_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf401, primals_883, primals_884, primals_885, primals_886, buf398, buf402, 2048, grid=grid(2048), stream=stream0)
        del primals_886
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf402, primals_967, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 4, 8, 8), (256, 64, 8, 1))
        buf436 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf435, primals_968, primals_969, primals_970, primals_971, buf436, 1024, grid=grid(1024), stream=stream0)
        del primals_971
        # Topologically Sorted Source Nodes: [out_488], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf368, primals_887, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 16, 4, 4), (256, 16, 4, 1))
        buf404 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_489, out_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf403, primals_888, primals_889, primals_890, primals_891, buf404, 1024, grid=grid(1024), stream=stream0)
        del primals_891
        # Topologically Sorted Source Nodes: [out_491], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_892, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (4, 16, 4, 4), (256, 16, 4, 1))
        buf406 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_492, out_493, out_494], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf405, primals_893, primals_894, primals_895, primals_896, buf368, buf406, 1024, grid=grid(1024), stream=stream0)
        del primals_896
        # Topologically Sorted Source Nodes: [out_495], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_897, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (4, 16, 4, 4), (256, 16, 4, 1))
        buf408 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_496, out_497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf407, primals_898, primals_899, primals_900, primals_901, buf408, 1024, grid=grid(1024), stream=stream0)
        del primals_901
        # Topologically Sorted Source Nodes: [out_498], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, primals_902, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 16, 4, 4), (256, 16, 4, 1))
        buf410 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_499, out_500, out_501], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf409, primals_903, primals_904, primals_905, primals_906, buf406, buf410, 1024, grid=grid(1024), stream=stream0)
        del primals_906
        # Topologically Sorted Source Nodes: [out_502], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_907, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 16, 4, 4), (256, 16, 4, 1))
        buf412 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_503, out_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf411, primals_908, primals_909, primals_910, primals_911, buf412, 1024, grid=grid(1024), stream=stream0)
        del primals_911
        # Topologically Sorted Source Nodes: [out_505], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_912, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (4, 16, 4, 4), (256, 16, 4, 1))
        buf414 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_506, out_507, out_508], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf413, primals_913, primals_914, primals_915, primals_916, buf410, buf414, 1024, grid=grid(1024), stream=stream0)
        del primals_916
        # Topologically Sorted Source Nodes: [out_509], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, primals_917, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (4, 16, 4, 4), (256, 16, 4, 1))
        buf416 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_510, out_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf415, primals_918, primals_919, primals_920, primals_921, buf416, 1024, grid=grid(1024), stream=stream0)
        del primals_921
        # Topologically Sorted Source Nodes: [out_512], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_922, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 16, 4, 4), (256, 16, 4, 1))
        buf418 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_513, out_514, out_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf417, primals_923, primals_924, primals_925, primals_926, buf414, buf418, 1024, grid=grid(1024), stream=stream0)
        del primals_926
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf418, primals_972, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 4, 4, 4), (64, 16, 4, 1))
        buf440 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf439, primals_973, primals_974, primals_975, primals_976, buf440, 256, grid=grid(256), stream=stream0)
        del primals_976
        # Topologically Sorted Source Nodes: [out_516], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf370, primals_927, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (4, 32, 2, 2), (128, 4, 2, 1))
        buf420 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_517, out_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf419, primals_928, primals_929, primals_930, primals_931, buf420, 512, grid=grid(512), stream=stream0)
        del primals_931
        # Topologically Sorted Source Nodes: [out_519], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, primals_932, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 32, 2, 2), (128, 4, 2, 1))
        buf422 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_520, out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf421, primals_933, primals_934, primals_935, primals_936, buf370, buf422, 512, grid=grid(512), stream=stream0)
        del primals_936
        # Topologically Sorted Source Nodes: [out_523], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_937, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 32, 2, 2), (128, 4, 2, 1))
        buf424 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_524, out_525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf423, primals_938, primals_939, primals_940, primals_941, buf424, 512, grid=grid(512), stream=stream0)
        del primals_941
        # Topologically Sorted Source Nodes: [out_526], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_942, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 32, 2, 2), (128, 4, 2, 1))
        buf426 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_527, out_528, out_529], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf425, primals_943, primals_944, primals_945, primals_946, buf422, buf426, 512, grid=grid(512), stream=stream0)
        del primals_946
        # Topologically Sorted Source Nodes: [out_530], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_947, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 32, 2, 2), (128, 4, 2, 1))
        buf428 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_531, out_532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf427, primals_948, primals_949, primals_950, primals_951, buf428, 512, grid=grid(512), stream=stream0)
        del primals_951
        # Topologically Sorted Source Nodes: [out_533], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_952, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 32, 2, 2), (128, 4, 2, 1))
        buf430 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_534, out_535, out_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf429, primals_953, primals_954, primals_955, primals_956, buf426, buf430, 512, grid=grid(512), stream=stream0)
        del primals_956
        # Topologically Sorted Source Nodes: [out_537], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_957, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 32, 2, 2), (128, 4, 2, 1))
        buf432 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_538, out_539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf431, primals_958, primals_959, primals_960, primals_961, buf432, 512, grid=grid(512), stream=stream0)
        del primals_961
        # Topologically Sorted Source Nodes: [out_540], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_962, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (4, 32, 2, 2), (128, 4, 2, 1))
        buf434 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_541, out_542, out_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf433, primals_963, primals_964, primals_965, primals_966, buf430, buf434, 512, grid=grid(512), stream=stream0)
        del primals_966
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf434, primals_977, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (4, 4, 2, 2), (16, 4, 2, 1))
        buf444 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf443, primals_978, primals_979, primals_980, primals_981, buf444, 64, grid=grid(64), stream=stream0)
        del primals_981
        buf445 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf445, 16, grid=grid(16), stream=stream0)
        buf446 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_31.run(buf446, 16, grid=grid(16), stream=stream0)
        buf447 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate, interpolate_15], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf447, 16, grid=grid(16), stream=stream0)
        buf448 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_31.run(buf448, 16, grid=grid(16), stream=stream0)
        buf449 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate, interpolate_15], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_32.run(buf449, 16, grid=grid(16), stream=stream0)
        buf451 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_15], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_32.run(buf451, 16, grid=grid(16), stream=stream0)
        buf386 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf437 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf438 = buf437; del buf437  # reuse
        buf442 = buf438; del buf438  # reuse
        buf452 = buf442; del buf442  # reuse
        buf453 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [out_457, out_458, out_459, interpolate_13, y_26, interpolate_14, y_27, interpolate_15, y_28, residual_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33.run(buf453, buf385, primals_843, primals_844, primals_845, primals_846, buf382, buf68, buf70, buf436, buf71, buf72, buf69, buf74, buf135, buf137, buf440, buf138, buf139, buf136, buf141, buf445, buf447, buf444, buf448, buf449, buf446, buf451, buf386, 4096, grid=grid(4096), stream=stream0)
        del primals_846
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf386, primals_982, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf418, primals_987, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (4, 8, 4, 4), (128, 16, 4, 1))
        buf456 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf455, primals_988, primals_989, primals_990, primals_991, buf456, 512, grid=grid(512), stream=stream0)
        del primals_991
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf434, primals_992, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 8, 2, 2), (32, 4, 2, 1))
        buf460 = empty_strided_cuda((4, 8, 2, 2), (32, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf459, primals_993, primals_994, primals_995, primals_996, buf460, 128, grid=grid(128), stream=stream0)
        del primals_996
        buf461 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_35.run(buf461, 8, grid=grid(8), stream=stream0)
        buf462 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_36.run(buf462, 8, grid=grid(8), stream=stream0)
        buf463 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_3, interpolate_17], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_35.run(buf463, 8, grid=grid(8), stream=stream0)
        buf464 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_36.run(buf464, 8, grid=grid(8), stream=stream0)
        buf465 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_3, interpolate_17], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_37.run(buf465, 8, grid=grid(8), stream=stream0)
        buf467 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_17], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_37.run(buf467, 8, grid=grid(8), stream=stream0)
        buf457 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf458 = buf457; del buf457  # reuse
        buf468 = buf458; del buf458  # reuse
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [input_86, y_29, interpolate_16, y_30, interpolate_17, y_31, residual_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38.run(buf469, buf147, buf149, buf456, buf150, buf151, buf454, primals_983, primals_984, primals_985, primals_986, buf402, buf148, buf153, buf461, buf463, buf460, buf464, buf465, buf462, buf467, 2048, grid=grid(2048), stream=stream0)
        del primals_986
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf386, primals_997, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 4, 8, 8), (256, 64, 8, 1))
        buf471 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf470, primals_998, primals_999, primals_1000, primals_1001, buf471, 1024, grid=grid(1024), stream=stream0)
        del primals_1001
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_1002, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf402, primals_1007, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf434, primals_1012, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 16, 2, 2), (64, 4, 2, 1))
        buf476 = reinterpret_tensor(buf440, (4, 16, 2, 2), (64, 4, 2, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf475, primals_1013, primals_1014, primals_1015, primals_1016, buf476, 256, grid=grid(256), stream=stream0)
        del primals_1016
        buf477 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(buf477, 4, grid=grid(4), stream=stream0)
        buf478 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_41.run(buf478, 4, grid=grid(4), stream=stream0)
        buf479 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(buf479, 4, grid=grid(4), stream=stream0)
        buf480 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_41.run(buf480, 4, grid=grid(4), stream=stream0)
        buf481 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_42.run(buf481, 4, grid=grid(4), stream=stream0)
        buf483 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_18], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_42.run(buf483, 4, grid=grid(4), stream=stream0)
        buf474 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf484 = buf474; del buf474  # reuse
        buf485 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [input_95, input_97, y_32, y_33, interpolate_18, y_34, residual_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43.run(buf485, buf472, primals_1003, primals_1004, primals_1005, primals_1006, buf473, primals_1008, primals_1009, primals_1010, primals_1011, buf477, buf479, buf476, buf480, buf481, buf418, buf478, buf483, 1024, grid=grid(1024), stream=stream0)
        del primals_1006
        del primals_1011
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf386, primals_1017, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 4, 8, 8), (256, 64, 8, 1))
        buf487 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf486, primals_1018, primals_1019, primals_1020, primals_1021, buf487, 1024, grid=grid(1024), stream=stream0)
        del primals_1021
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_1022, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 4, 4, 4), (64, 16, 4, 1))
        buf489 = reinterpret_tensor(buf476, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf488, primals_1023, primals_1024, primals_1025, primals_1026, buf489, 256, grid=grid(256), stream=stream0)
        del primals_1026
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_1027, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf402, primals_1032, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 8, 4, 4), (128, 16, 4, 1))
        buf492 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [input_109, input_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf491, primals_1033, primals_1034, primals_1035, primals_1036, buf492, 512, grid=grid(512), stream=stream0)
        del primals_1036
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf493 = extern_kernels.convolution(buf492, primals_1037, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_113], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf418, primals_1042, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 32, 2, 2), (128, 4, 2, 1))
        buf494 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        buf496 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [input_107, input_112, y_35, input_114, y_36, y_37, residual_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf496, buf490, primals_1028, primals_1029, primals_1030, primals_1031, buf493, primals_1038, primals_1039, primals_1040, primals_1041, buf495, primals_1043, primals_1044, primals_1045, primals_1046, buf434, 512, grid=grid(512), stream=stream0)
        del primals_1031
        del primals_1041
        del primals_1046
        # Topologically Sorted Source Nodes: [out_544], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf453, primals_1047, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf498 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_545, out_546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf497, primals_1048, primals_1049, primals_1050, primals_1051, buf498, 4096, grid=grid(4096), stream=stream0)
        del primals_1051
        # Topologically Sorted Source Nodes: [out_547], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, primals_1052, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf500 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_548, out_549, out_550], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf499, primals_1053, primals_1054, primals_1055, primals_1056, buf453, buf500, 4096, grid=grid(4096), stream=stream0)
        del primals_1056
        # Topologically Sorted Source Nodes: [out_551], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_1057, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf502 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_552, out_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf501, primals_1058, primals_1059, primals_1060, primals_1061, buf502, 4096, grid=grid(4096), stream=stream0)
        del primals_1061
        # Topologically Sorted Source Nodes: [out_554], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, primals_1062, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf504 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_555, out_556, out_557], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf503, primals_1063, primals_1064, primals_1065, primals_1066, buf500, buf504, 4096, grid=grid(4096), stream=stream0)
        del primals_1066
        # Topologically Sorted Source Nodes: [out_558], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_1067, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf506 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_559, out_560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf505, primals_1068, primals_1069, primals_1070, primals_1071, buf506, 4096, grid=grid(4096), stream=stream0)
        del primals_1071
        # Topologically Sorted Source Nodes: [out_561], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_1072, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf508 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_562, out_563, out_564], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf507, primals_1073, primals_1074, primals_1075, primals_1076, buf504, buf508, 4096, grid=grid(4096), stream=stream0)
        del primals_1076
        # Topologically Sorted Source Nodes: [out_565], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, primals_1077, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf510 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_566, out_567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf509, primals_1078, primals_1079, primals_1080, primals_1081, buf510, 4096, grid=grid(4096), stream=stream0)
        del primals_1081
        # Topologically Sorted Source Nodes: [out_568], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf510, primals_1082, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_572], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf469, primals_1087, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 8, 8, 8), (512, 64, 8, 1))
        buf514 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_573, out_574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf513, primals_1088, primals_1089, primals_1090, primals_1091, buf514, 2048, grid=grid(2048), stream=stream0)
        del primals_1091
        # Topologically Sorted Source Nodes: [out_575], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_1092, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (4, 8, 8, 8), (512, 64, 8, 1))
        buf516 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_576, out_577, out_578], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf515, primals_1093, primals_1094, primals_1095, primals_1096, buf469, buf516, 2048, grid=grid(2048), stream=stream0)
        del primals_1096
        # Topologically Sorted Source Nodes: [out_579], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_1097, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (4, 8, 8, 8), (512, 64, 8, 1))
        buf518 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_580, out_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf517, primals_1098, primals_1099, primals_1100, primals_1101, buf518, 2048, grid=grid(2048), stream=stream0)
        del primals_1101
        # Topologically Sorted Source Nodes: [out_582], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_1102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (4, 8, 8, 8), (512, 64, 8, 1))
        buf520 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_583, out_584, out_585], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf519, primals_1103, primals_1104, primals_1105, primals_1106, buf516, buf520, 2048, grid=grid(2048), stream=stream0)
        del primals_1106
        # Topologically Sorted Source Nodes: [out_586], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf520, primals_1107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (4, 8, 8, 8), (512, 64, 8, 1))
        buf522 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_587, out_588], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf521, primals_1108, primals_1109, primals_1110, primals_1111, buf522, 2048, grid=grid(2048), stream=stream0)
        del primals_1111
        # Topologically Sorted Source Nodes: [out_589], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_1112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 8, 8, 8), (512, 64, 8, 1))
        buf524 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_590, out_591, out_592], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf523, primals_1113, primals_1114, primals_1115, primals_1116, buf520, buf524, 2048, grid=grid(2048), stream=stream0)
        del primals_1116
        # Topologically Sorted Source Nodes: [out_593], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, primals_1117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (4, 8, 8, 8), (512, 64, 8, 1))
        buf526 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_594, out_595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf525, primals_1118, primals_1119, primals_1120, primals_1121, buf526, 2048, grid=grid(2048), stream=stream0)
        del primals_1121
        # Topologically Sorted Source Nodes: [out_596], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_1122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (4, 8, 8, 8), (512, 64, 8, 1))
        buf528 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_597, out_598, out_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf527, primals_1123, primals_1124, primals_1125, primals_1126, buf524, buf528, 2048, grid=grid(2048), stream=stream0)
        del primals_1126
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf528, primals_1207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf561, (4, 4, 8, 8), (256, 64, 8, 1))
        buf562 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf561, primals_1208, primals_1209, primals_1210, primals_1211, buf562, 1024, grid=grid(1024), stream=stream0)
        del primals_1211
        # Topologically Sorted Source Nodes: [out_600], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf485, primals_1127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (4, 16, 4, 4), (256, 16, 4, 1))
        buf530 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_601, out_602], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf529, primals_1128, primals_1129, primals_1130, primals_1131, buf530, 1024, grid=grid(1024), stream=stream0)
        del primals_1131
        # Topologically Sorted Source Nodes: [out_603], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_1132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (4, 16, 4, 4), (256, 16, 4, 1))
        buf532 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_604, out_605, out_606], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf531, primals_1133, primals_1134, primals_1135, primals_1136, buf485, buf532, 1024, grid=grid(1024), stream=stream0)
        del primals_1136
        # Topologically Sorted Source Nodes: [out_607], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, primals_1137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (4, 16, 4, 4), (256, 16, 4, 1))
        buf534 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_608, out_609], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf533, primals_1138, primals_1139, primals_1140, primals_1141, buf534, 1024, grid=grid(1024), stream=stream0)
        del primals_1141
        # Topologically Sorted Source Nodes: [out_610], Original ATen: [aten.convolution]
        buf535 = extern_kernels.convolution(buf534, primals_1142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf535, (4, 16, 4, 4), (256, 16, 4, 1))
        buf536 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_611, out_612, out_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf535, primals_1143, primals_1144, primals_1145, primals_1146, buf532, buf536, 1024, grid=grid(1024), stream=stream0)
        del primals_1146
        # Topologically Sorted Source Nodes: [out_614], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_1147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (4, 16, 4, 4), (256, 16, 4, 1))
        buf538 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_615, out_616], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf537, primals_1148, primals_1149, primals_1150, primals_1151, buf538, 1024, grid=grid(1024), stream=stream0)
        del primals_1151
        # Topologically Sorted Source Nodes: [out_617], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_1152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (4, 16, 4, 4), (256, 16, 4, 1))
        buf540 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_618, out_619, out_620], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf539, primals_1153, primals_1154, primals_1155, primals_1156, buf536, buf540, 1024, grid=grid(1024), stream=stream0)
        del primals_1156
        # Topologically Sorted Source Nodes: [out_621], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_1157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (4, 16, 4, 4), (256, 16, 4, 1))
        buf542 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_622, out_623], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf541, primals_1158, primals_1159, primals_1160, primals_1161, buf542, 1024, grid=grid(1024), stream=stream0)
        del primals_1161
        # Topologically Sorted Source Nodes: [out_624], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_1162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 16, 4, 4), (256, 16, 4, 1))
        buf544 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_625, out_626, out_627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf543, primals_1163, primals_1164, primals_1165, primals_1166, buf540, buf544, 1024, grid=grid(1024), stream=stream0)
        del primals_1166
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf544, primals_1212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (4, 4, 4, 4), (64, 16, 4, 1))
        buf566 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf565, primals_1213, primals_1214, primals_1215, primals_1216, buf566, 256, grid=grid(256), stream=stream0)
        del primals_1216
        # Topologically Sorted Source Nodes: [out_628], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf496, primals_1167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (4, 32, 2, 2), (128, 4, 2, 1))
        buf546 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_629, out_630], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf545, primals_1168, primals_1169, primals_1170, primals_1171, buf546, 512, grid=grid(512), stream=stream0)
        del primals_1171
        # Topologically Sorted Source Nodes: [out_631], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_1172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (4, 32, 2, 2), (128, 4, 2, 1))
        buf548 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_632, out_633, out_634], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf547, primals_1173, primals_1174, primals_1175, primals_1176, buf496, buf548, 512, grid=grid(512), stream=stream0)
        del primals_1176
        # Topologically Sorted Source Nodes: [out_635], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_1177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (4, 32, 2, 2), (128, 4, 2, 1))
        buf550 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_636, out_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf549, primals_1178, primals_1179, primals_1180, primals_1181, buf550, 512, grid=grid(512), stream=stream0)
        del primals_1181
        # Topologically Sorted Source Nodes: [out_638], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_1182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 32, 2, 2), (128, 4, 2, 1))
        buf552 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_639, out_640, out_641], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf551, primals_1183, primals_1184, primals_1185, primals_1186, buf548, buf552, 512, grid=grid(512), stream=stream0)
        del primals_1186
        # Topologically Sorted Source Nodes: [out_642], Original ATen: [aten.convolution]
        buf553 = extern_kernels.convolution(buf552, primals_1187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf553, (4, 32, 2, 2), (128, 4, 2, 1))
        buf554 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_643, out_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf553, primals_1188, primals_1189, primals_1190, primals_1191, buf554, 512, grid=grid(512), stream=stream0)
        del primals_1191
        # Topologically Sorted Source Nodes: [out_645], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, primals_1192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (4, 32, 2, 2), (128, 4, 2, 1))
        buf556 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_646, out_647, out_648], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf555, primals_1193, primals_1194, primals_1195, primals_1196, buf552, buf556, 512, grid=grid(512), stream=stream0)
        del primals_1196
        # Topologically Sorted Source Nodes: [out_649], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_1197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (4, 32, 2, 2), (128, 4, 2, 1))
        buf558 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_650, out_651], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf557, primals_1198, primals_1199, primals_1200, primals_1201, buf558, 512, grid=grid(512), stream=stream0)
        del primals_1201
        # Topologically Sorted Source Nodes: [out_652], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf558, primals_1202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (4, 32, 2, 2), (128, 4, 2, 1))
        buf560 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_653, out_654, out_655], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf559, primals_1203, primals_1204, primals_1205, primals_1206, buf556, buf560, 512, grid=grid(512), stream=stream0)
        del primals_1206
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf560, primals_1217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (4, 4, 2, 2), (16, 4, 2, 1))
        buf570 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf569, primals_1218, primals_1219, primals_1220, primals_1221, buf570, 64, grid=grid(64), stream=stream0)
        del primals_1221
        buf512 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf563 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf564 = buf563; del buf563  # reuse
        buf568 = buf564; del buf564  # reuse
        buf572 = buf568; del buf568  # reuse
        buf573 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [out_569, out_570, out_571, interpolate_19, y_38, interpolate_20, y_39, interpolate_21, y_40, residual_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_33.run(buf573, buf511, primals_1083, primals_1084, primals_1085, primals_1086, buf508, buf68, buf70, buf562, buf71, buf72, buf69, buf74, buf135, buf137, buf566, buf138, buf139, buf136, buf141, buf445, buf447, buf570, buf448, buf449, buf446, buf451, buf512, 4096, grid=grid(4096), stream=stream0)
        del primals_1086
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf512, primals_1222, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf544, primals_1227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (4, 8, 4, 4), (128, 16, 4, 1))
        buf576 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf575, primals_1228, primals_1229, primals_1230, primals_1231, buf576, 512, grid=grid(512), stream=stream0)
        del primals_1231
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf560, primals_1232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 8, 2, 2), (32, 4, 2, 1))
        buf580 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf579, primals_1233, primals_1234, primals_1235, primals_1236, buf580, 128, grid=grid(128), stream=stream0)
        del primals_1236
        buf577 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf578 = buf577; del buf577  # reuse
        buf582 = buf578; del buf578  # reuse
        buf583 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [input_122, y_41, interpolate_22, y_42, interpolate_23, y_43, residual_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_38.run(buf583, buf147, buf149, buf576, buf150, buf151, buf574, primals_1223, primals_1224, primals_1225, primals_1226, buf528, buf148, buf153, buf461, buf463, buf580, buf464, buf465, buf462, buf467, 2048, grid=grid(2048), stream=stream0)
        del primals_1226
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf512, primals_1237, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 4, 8, 8), (256, 64, 8, 1))
        buf585 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf584, primals_1238, primals_1239, primals_1240, primals_1241, buf585, 1024, grid=grid(1024), stream=stream0)
        del primals_1241
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_1242, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_132], Original ATen: [aten.convolution]
        buf587 = extern_kernels.convolution(buf528, primals_1247, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf587, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf560, primals_1252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (4, 16, 2, 2), (64, 4, 2, 1))
        buf590 = reinterpret_tensor(buf566, (4, 16, 2, 2), (64, 4, 2, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf589, primals_1253, primals_1254, primals_1255, primals_1256, buf590, 256, grid=grid(256), stream=stream0)
        del primals_1256
        buf588 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf592 = buf588; del buf588  # reuse
        buf593 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [input_131, input_133, y_44, y_45, interpolate_24, y_46, residual_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_43.run(buf593, buf586, primals_1243, primals_1244, primals_1245, primals_1246, buf587, primals_1248, primals_1249, primals_1250, primals_1251, buf477, buf479, buf590, buf480, buf481, buf544, buf478, buf483, 1024, grid=grid(1024), stream=stream0)
        del primals_1246
        del primals_1251
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf512, primals_1257, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (4, 4, 8, 8), (256, 64, 8, 1))
        buf595 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_137, input_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf594, primals_1258, primals_1259, primals_1260, primals_1261, buf595, 1024, grid=grid(1024), stream=stream0)
        del primals_1261
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_1262, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 4, 4, 4), (64, 16, 4, 1))
        buf597 = reinterpret_tensor(buf590, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [input_140, input_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf596, primals_1263, primals_1264, primals_1265, primals_1266, buf597, 256, grid=grid(256), stream=stream0)
        del primals_1266
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf597, primals_1267, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf528, primals_1272, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (4, 8, 4, 4), (128, 16, 4, 1))
        buf600 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [input_145, input_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf599, primals_1273, primals_1274, primals_1275, primals_1276, buf600, 512, grid=grid(512), stream=stream0)
        del primals_1276
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_1277, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf544, primals_1282, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (4, 32, 2, 2), (128, 4, 2, 1))
        buf602 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        buf604 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [input_143, input_148, y_47, input_150, y_48, y_49, residual_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_46.run(buf604, buf598, primals_1268, primals_1269, primals_1270, primals_1271, buf601, primals_1278, primals_1279, primals_1280, primals_1281, buf603, primals_1283, primals_1284, primals_1285, primals_1286, buf560, 512, grid=grid(512), stream=stream0)
        del primals_1271
        del primals_1281
        del primals_1286
        # Topologically Sorted Source Nodes: [out_656], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf573, primals_1287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf606 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_657, out_658], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf605, primals_1288, primals_1289, primals_1290, primals_1291, buf606, 4096, grid=grid(4096), stream=stream0)
        del primals_1291
        # Topologically Sorted Source Nodes: [out_659], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, primals_1292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf608 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_660, out_661, out_662], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf607, primals_1293, primals_1294, primals_1295, primals_1296, buf573, buf608, 4096, grid=grid(4096), stream=stream0)
        del primals_1296
        # Topologically Sorted Source Nodes: [out_663], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf608, primals_1297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf610 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_664, out_665], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf609, primals_1298, primals_1299, primals_1300, primals_1301, buf610, 4096, grid=grid(4096), stream=stream0)
        del primals_1301
        # Topologically Sorted Source Nodes: [out_666], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_1302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf612 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_667, out_668, out_669], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf611, primals_1303, primals_1304, primals_1305, primals_1306, buf608, buf612, 4096, grid=grid(4096), stream=stream0)
        del primals_1306
        # Topologically Sorted Source Nodes: [out_670], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, primals_1307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf614 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_671, out_672], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf613, primals_1308, primals_1309, primals_1310, primals_1311, buf614, 4096, grid=grid(4096), stream=stream0)
        del primals_1311
        # Topologically Sorted Source Nodes: [out_673], Original ATen: [aten.convolution]
        buf615 = extern_kernels.convolution(buf614, primals_1312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf615, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf616 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_674, out_675, out_676], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf615, primals_1313, primals_1314, primals_1315, primals_1316, buf612, buf616, 4096, grid=grid(4096), stream=stream0)
        del primals_1316
        # Topologically Sorted Source Nodes: [out_677], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_1317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf618 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_678, out_679], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf617, primals_1318, primals_1319, primals_1320, primals_1321, buf618, 4096, grid=grid(4096), stream=stream0)
        del primals_1321
        # Topologically Sorted Source Nodes: [out_680], Original ATen: [aten.convolution]
        buf619 = extern_kernels.convolution(buf618, primals_1322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf619, (4, 4, 16, 16), (1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_684], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf583, primals_1327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (4, 8, 8, 8), (512, 64, 8, 1))
        buf622 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_685, out_686], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf621, primals_1328, primals_1329, primals_1330, primals_1331, buf622, 2048, grid=grid(2048), stream=stream0)
        del primals_1331
        # Topologically Sorted Source Nodes: [out_687], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, primals_1332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (4, 8, 8, 8), (512, 64, 8, 1))
        buf624 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_688, out_689, out_690], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf623, primals_1333, primals_1334, primals_1335, primals_1336, buf583, buf624, 2048, grid=grid(2048), stream=stream0)
        del primals_1336
        # Topologically Sorted Source Nodes: [out_691], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, primals_1337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf625, (4, 8, 8, 8), (512, 64, 8, 1))
        buf626 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_692, out_693], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf625, primals_1338, primals_1339, primals_1340, primals_1341, buf626, 2048, grid=grid(2048), stream=stream0)
        del primals_1341
        # Topologically Sorted Source Nodes: [out_694], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_1342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (4, 8, 8, 8), (512, 64, 8, 1))
        buf628 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_695, out_696, out_697], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf627, primals_1343, primals_1344, primals_1345, primals_1346, buf624, buf628, 2048, grid=grid(2048), stream=stream0)
        del primals_1346
        # Topologically Sorted Source Nodes: [out_698], Original ATen: [aten.convolution]
        buf629 = extern_kernels.convolution(buf628, primals_1347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf629, (4, 8, 8, 8), (512, 64, 8, 1))
        buf630 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_699, out_700], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf629, primals_1348, primals_1349, primals_1350, primals_1351, buf630, 2048, grid=grid(2048), stream=stream0)
        del primals_1351
        # Topologically Sorted Source Nodes: [out_701], Original ATen: [aten.convolution]
        buf631 = extern_kernels.convolution(buf630, primals_1352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (4, 8, 8, 8), (512, 64, 8, 1))
        buf632 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_702, out_703, out_704], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf631, primals_1353, primals_1354, primals_1355, primals_1356, buf628, buf632, 2048, grid=grid(2048), stream=stream0)
        del primals_1356
        # Topologically Sorted Source Nodes: [out_705], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_1357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (4, 8, 8, 8), (512, 64, 8, 1))
        buf634 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_706, out_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf633, primals_1358, primals_1359, primals_1360, primals_1361, buf634, 2048, grid=grid(2048), stream=stream0)
        del primals_1361
        # Topologically Sorted Source Nodes: [out_708], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf634, primals_1362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (4, 8, 8, 8), (512, 64, 8, 1))
        buf636 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_709, out_710, out_711], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf635, primals_1363, primals_1364, primals_1365, primals_1366, buf632, buf636, 2048, grid=grid(2048), stream=stream0)
        del primals_1366
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf636, primals_1447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (4, 4, 8, 8), (256, 64, 8, 1))
        buf670 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf669, primals_1448, primals_1449, primals_1450, primals_1451, buf670, 1024, grid=grid(1024), stream=stream0)
        del primals_1451
        # Topologically Sorted Source Nodes: [out_712], Original ATen: [aten.convolution]
        buf637 = extern_kernels.convolution(buf593, primals_1367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf637, (4, 16, 4, 4), (256, 16, 4, 1))
        buf638 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_713, out_714], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf637, primals_1368, primals_1369, primals_1370, primals_1371, buf638, 1024, grid=grid(1024), stream=stream0)
        del primals_1371
        # Topologically Sorted Source Nodes: [out_715], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_1372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (4, 16, 4, 4), (256, 16, 4, 1))
        buf640 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_716, out_717, out_718], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf639, primals_1373, primals_1374, primals_1375, primals_1376, buf593, buf640, 1024, grid=grid(1024), stream=stream0)
        del primals_1376
        # Topologically Sorted Source Nodes: [out_719], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_1377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (4, 16, 4, 4), (256, 16, 4, 1))
        buf642 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_720, out_721], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf641, primals_1378, primals_1379, primals_1380, primals_1381, buf642, 1024, grid=grid(1024), stream=stream0)
        del primals_1381
        # Topologically Sorted Source Nodes: [out_722], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_1382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (4, 16, 4, 4), (256, 16, 4, 1))
        buf644 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_723, out_724, out_725], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf643, primals_1383, primals_1384, primals_1385, primals_1386, buf640, buf644, 1024, grid=grid(1024), stream=stream0)
        del primals_1386
        # Topologically Sorted Source Nodes: [out_726], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, primals_1387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (4, 16, 4, 4), (256, 16, 4, 1))
        buf646 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_727, out_728], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf645, primals_1388, primals_1389, primals_1390, primals_1391, buf646, 1024, grid=grid(1024), stream=stream0)
        del primals_1391
        # Topologically Sorted Source Nodes: [out_729], Original ATen: [aten.convolution]
        buf647 = extern_kernels.convolution(buf646, primals_1392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf647, (4, 16, 4, 4), (256, 16, 4, 1))
        buf648 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_730, out_731, out_732], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf647, primals_1393, primals_1394, primals_1395, primals_1396, buf644, buf648, 1024, grid=grid(1024), stream=stream0)
        del primals_1396
        # Topologically Sorted Source Nodes: [out_733], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_1397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (4, 16, 4, 4), (256, 16, 4, 1))
        buf650 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_734, out_735], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf649, primals_1398, primals_1399, primals_1400, primals_1401, buf650, 1024, grid=grid(1024), stream=stream0)
        del primals_1401
        # Topologically Sorted Source Nodes: [out_736], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, primals_1402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (4, 16, 4, 4), (256, 16, 4, 1))
        buf652 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_737, out_738, out_739], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf651, primals_1403, primals_1404, primals_1405, primals_1406, buf648, buf652, 1024, grid=grid(1024), stream=stream0)
        del primals_1406
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf652, primals_1452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (4, 4, 4, 4), (64, 16, 4, 1))
        buf674 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf673, primals_1453, primals_1454, primals_1455, primals_1456, buf674, 256, grid=grid(256), stream=stream0)
        del primals_1456
        # Topologically Sorted Source Nodes: [out_740], Original ATen: [aten.convolution]
        buf653 = extern_kernels.convolution(buf604, primals_1407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf653, (4, 32, 2, 2), (128, 4, 2, 1))
        buf654 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_741, out_742], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf653, primals_1408, primals_1409, primals_1410, primals_1411, buf654, 512, grid=grid(512), stream=stream0)
        del primals_1411
        # Topologically Sorted Source Nodes: [out_743], Original ATen: [aten.convolution]
        buf655 = extern_kernels.convolution(buf654, primals_1412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (4, 32, 2, 2), (128, 4, 2, 1))
        buf656 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_744, out_745, out_746], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf655, primals_1413, primals_1414, primals_1415, primals_1416, buf604, buf656, 512, grid=grid(512), stream=stream0)
        del primals_1416
        # Topologically Sorted Source Nodes: [out_747], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf656, primals_1417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (4, 32, 2, 2), (128, 4, 2, 1))
        buf658 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_748, out_749], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf657, primals_1418, primals_1419, primals_1420, primals_1421, buf658, 512, grid=grid(512), stream=stream0)
        del primals_1421
        # Topologically Sorted Source Nodes: [out_750], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf658, primals_1422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (4, 32, 2, 2), (128, 4, 2, 1))
        buf660 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_751, out_752, out_753], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf659, primals_1423, primals_1424, primals_1425, primals_1426, buf656, buf660, 512, grid=grid(512), stream=stream0)
        del primals_1426
        # Topologically Sorted Source Nodes: [out_754], Original ATen: [aten.convolution]
        buf661 = extern_kernels.convolution(buf660, primals_1427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf661, (4, 32, 2, 2), (128, 4, 2, 1))
        buf662 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_755, out_756], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf661, primals_1428, primals_1429, primals_1430, primals_1431, buf662, 512, grid=grid(512), stream=stream0)
        del primals_1431
        # Topologically Sorted Source Nodes: [out_757], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, primals_1432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (4, 32, 2, 2), (128, 4, 2, 1))
        buf664 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_758, out_759, out_760], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf663, primals_1433, primals_1434, primals_1435, primals_1436, buf660, buf664, 512, grid=grid(512), stream=stream0)
        del primals_1436
        # Topologically Sorted Source Nodes: [out_761], Original ATen: [aten.convolution]
        buf665 = extern_kernels.convolution(buf664, primals_1437, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf665, (4, 32, 2, 2), (128, 4, 2, 1))
        buf666 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_762, out_763], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf665, primals_1438, primals_1439, primals_1440, primals_1441, buf666, 512, grid=grid(512), stream=stream0)
        del primals_1441
        # Topologically Sorted Source Nodes: [out_764], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, primals_1442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (4, 32, 2, 2), (128, 4, 2, 1))
        buf668 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_765, out_766, out_767], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf667, primals_1443, primals_1444, primals_1445, primals_1446, buf664, buf668, 512, grid=grid(512), stream=stream0)
        del primals_1446
        # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.convolution]
        buf677 = extern_kernels.convolution(buf668, primals_1457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf677, (4, 4, 2, 2), (16, 4, 2, 1))
        buf678 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf677, primals_1458, primals_1459, primals_1460, primals_1461, buf678, 64, grid=grid(64), stream=stream0)
        del primals_1461
        buf620 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf671 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        buf672 = buf671; del buf671  # reuse
        buf676 = buf672; del buf672  # reuse
        buf680 = buf676; del buf676  # reuse
        buf755 = empty_strided_cuda((4, 4, 16, 16), (1024, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_681, out_682, out_683, interpolate_25, y_50, interpolate_26, y_51, interpolate_27, y_52, relu_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_47.run(buf680, buf619, primals_1323, primals_1324, primals_1325, primals_1326, buf616, buf68, buf70, buf670, buf71, buf72, buf69, buf74, buf135, buf137, buf674, buf138, buf139, buf136, buf141, buf445, buf447, buf678, buf448, buf449, buf446, buf451, buf620, buf755, 4096, grid=grid(4096), stream=stream0)
        del buf678
        del primals_1326
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.convolution]
        buf681 = extern_kernels.convolution(buf620, primals_1462, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf681, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf652, primals_1467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (4, 8, 4, 4), (128, 16, 4, 1))
        buf683 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf682, primals_1468, primals_1469, primals_1470, primals_1471, buf683, 512, grid=grid(512), stream=stream0)
        del primals_1471
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf668, primals_1472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (4, 8, 2, 2), (32, 4, 2, 1))
        buf687 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf686, primals_1473, primals_1474, primals_1475, primals_1476, buf687, 128, grid=grid(128), stream=stream0)
        del primals_1476
        buf684 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.float32)
        buf685 = buf684; del buf684  # reuse
        buf689 = buf685; del buf685  # reuse
        buf754 = empty_strided_cuda((4, 8, 8, 8), (512, 64, 8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_158, y_53, interpolate_28, y_54, interpolate_29, y_55, relu_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48.run(buf689, buf147, buf149, buf683, buf150, buf151, buf681, primals_1463, primals_1464, primals_1465, primals_1466, buf636, buf148, buf153, buf461, buf463, buf687, buf464, buf465, buf462, buf467, buf754, 2048, grid=grid(2048), stream=stream0)
        del buf687
        del primals_1466
        # Topologically Sorted Source Nodes: [input_163], Original ATen: [aten.convolution]
        buf690 = extern_kernels.convolution(buf620, primals_1477, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf690, (4, 4, 8, 8), (256, 64, 8, 1))
        buf691 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [input_164, input_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf690, primals_1478, primals_1479, primals_1480, primals_1481, buf691, 1024, grid=grid(1024), stream=stream0)
        del primals_1481
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf692 = extern_kernels.convolution(buf691, primals_1482, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf693 = extern_kernels.convolution(buf636, primals_1487, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf668, primals_1492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (4, 16, 2, 2), (64, 4, 2, 1))
        buf696 = reinterpret_tensor(buf674, (4, 16, 2, 2), (64, 4, 2, 1), 0); del buf674  # reuse
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf695, primals_1493, primals_1494, primals_1495, primals_1496, buf696, 256, grid=grid(256), stream=stream0)
        del primals_1496
        buf694 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf698 = buf694; del buf694  # reuse
        buf753 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_167, input_169, y_56, y_57, interpolate_30, y_58, relu_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index, aten.sub, aten.mul, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_49.run(buf698, buf692, primals_1483, primals_1484, primals_1485, primals_1486, buf693, primals_1488, primals_1489, primals_1490, primals_1491, buf477, buf479, buf696, buf480, buf481, buf652, buf478, buf483, buf753, 1024, grid=grid(1024), stream=stream0)
        del primals_1486
        del primals_1491
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf699 = extern_kernels.convolution(buf620, primals_1497, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf699, (4, 4, 8, 8), (256, 64, 8, 1))
        buf700 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_173, input_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf699, primals_1498, primals_1499, primals_1500, primals_1501, buf700, 1024, grid=grid(1024), stream=stream0)
        del primals_1501
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf701 = extern_kernels.convolution(buf700, primals_1502, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf701, (4, 4, 4, 4), (64, 16, 4, 1))
        buf702 = reinterpret_tensor(buf696, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf701, primals_1503, primals_1504, primals_1505, primals_1506, buf702, 256, grid=grid(256), stream=stream0)
        del primals_1506
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf703 = extern_kernels.convolution(buf702, primals_1507, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf704 = extern_kernels.convolution(buf636, primals_1512, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf704, (4, 8, 4, 4), (128, 16, 4, 1))
        buf705 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [input_181, input_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf704, primals_1513, primals_1514, primals_1515, primals_1516, buf705, 512, grid=grid(512), stream=stream0)
        del primals_1516
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_1517, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (4, 32, 2, 2), (128, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf708 = extern_kernels.convolution(buf652, primals_1522, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf708, (4, 32, 2, 2), (128, 4, 2, 1))
        buf707 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        buf709 = buf707; del buf707  # reuse
        buf752 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_179, input_184, y_59, input_186, y_60, y_61, relu_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_50.run(buf709, buf703, primals_1508, primals_1509, primals_1510, primals_1511, buf706, primals_1518, primals_1519, primals_1520, primals_1521, buf708, primals_1523, primals_1524, primals_1525, primals_1526, buf668, buf752, 512, grid=grid(512), stream=stream0)
        del primals_1511
        del primals_1521
        del primals_1526
        buf710 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu_261, x1], Original ATen: [aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_relu_sub_51.run(buf68, buf70, buf689, buf71, buf72, buf710, 8192, grid=grid(8192), stream=stream0)
        buf711 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu_263, x2], Original ATen: [aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_relu_sub_52.run(buf135, buf137, buf698, buf138, buf139, buf711, 16384, grid=grid(16384), stream=stream0)
        buf712 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_53.run(buf445, buf447, buf709, buf448, buf449, buf712, 32768, grid=grid(32768), stream=stream0)
        buf713 = empty_strided_cuda((4, 60, 16, 16), (15360, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feats], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf680, buf710, buf69, buf70, buf689, buf71, buf72, buf74, buf711, buf136, buf137, buf698, buf138, buf139, buf141, buf712, buf446, buf447, buf709, buf448, buf449, buf451, buf713, 61440, grid=grid(61440), stream=stream0)
        del buf689
        del buf698
        del buf709
        del buf711
        del buf712
        # Topologically Sorted Source Nodes: [input_187], Original ATen: [aten.convolution]
        buf714 = extern_kernels.convolution(buf713, primals_1527, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf714, (4, 60, 16, 16), (15360, 256, 16, 1))
        buf715 = buf714; del buf714  # reuse
        buf716 = empty_strided_cuda((4, 60, 16, 16), (15360, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_187, input_188, input_189], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55.run(buf715, primals_1528, primals_1529, primals_1530, primals_1531, primals_1532, buf716, 61440, grid=grid(61440), stream=stream0)
        del primals_1528
        del primals_1532
        # Topologically Sorted Source Nodes: [input_190], Original ATen: [aten.convolution]
        buf717 = extern_kernels.convolution(buf716, primals_1533, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf717, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf718 = buf717; del buf717  # reuse
        buf724 = reinterpret_tensor(buf680, (4, 4, 256), (1024, 256, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [input_190, probs_1], Original ATen: [aten.convolution, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_convolution_56.run(buf718, primals_1534, buf724, 16, 256, grid=grid(16), stream=stream0)
        del primals_1534
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf719 = extern_kernels.convolution(buf713, primals_1535, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf719, (4, 512, 16, 16), (131072, 256, 16, 1))
        buf720 = buf719; del buf719  # reuse
        buf721 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_191, input_192, input_193], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57.run(buf720, primals_1536, primals_1537, primals_1538, primals_1539, primals_1540, buf721, 524288, grid=grid(524288), stream=stream0)
        del primals_1536
        del primals_1540
        buf725 = reinterpret_tensor(buf710, (4, 4, 512), (2048, 512, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf724, reinterpret_tensor(buf721, (4, 256, 512), (131072, 1, 256), 0), out=buf725)
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf726 = extern_kernels.convolution(buf721, primals_1541, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf726, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf727 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf726, primals_1542, primals_1543, primals_1544, primals_1545, buf727, 262144, grid=grid(262144), stream=stream0)
        del primals_1545
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        buf728 = extern_kernels.convolution(buf727, primals_1546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf728, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf729 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf751 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_59.run(buf728, primals_1547, primals_1548, primals_1549, primals_1550, buf729, buf751, 262144, grid=grid(262144), stream=stream0)
        del primals_1550
        buf730 = empty_strided_cuda((4, 512, 4, 1), (2048, 4, 1, 1), torch.float32)
        buf735 = empty_strided_cuda((4, 512, 4, 1), (2048, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_200, input_206], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_60.run(buf725, buf730, buf735, 2048, 4, grid=grid(2048, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.convolution]
        buf731 = extern_kernels.convolution(buf730, primals_1551, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf731, (4, 256, 4, 1), (1024, 4, 1, 1))
        del buf730
        buf732 = empty_strided_cuda((4, 256, 4, 1), (1024, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_201, input_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_61.run(buf731, primals_1552, primals_1553, primals_1554, primals_1555, buf732, 4096, grid=grid(4096), stream=stream0)
        del primals_1555
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf733 = extern_kernels.convolution(buf732, primals_1556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf733, (4, 256, 4, 1), (1024, 4, 1, 1))
        buf734 = empty_strided_cuda((4, 256, 4, 1), (1024, 4, 1, 1), torch.float32)
        buf750 = empty_strided_cuda((4, 256, 4, 1), (1024, 4, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62.run(buf733, primals_1557, primals_1558, primals_1559, primals_1560, buf734, buf750, 4096, grid=grid(4096), stream=stream0)
        del primals_1560
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf736 = extern_kernels.convolution(buf735, primals_1561, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf736, (4, 256, 4, 1), (1024, 4, 1, 1))
        del buf735
        buf737 = empty_strided_cuda((4, 256, 4, 1), (1024, 4, 1, 1), torch.float32)
        buf749 = empty_strided_cuda((4, 256, 4, 1), (1024, 4, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_207, input_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_62.run(buf736, primals_1562, primals_1563, primals_1564, primals_1565, buf737, buf749, 4096, grid=grid(4096), stream=stream0)
        del primals_1565
        buf738 = empty_strided_cuda((4, 256, 4), (1024, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (4, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf734, (4, 256, 4), (1024, 4, 1), 0), out=buf738)
        buf739 = empty_strided_cuda((4, 256, 4), (1024, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_63.run(buf738, buf739, 4096, grid=grid(4096), stream=stream0)
        buf740 = buf738; del buf738  # reuse
        # Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_64.run(buf739, buf740, 4096, grid=grid(4096), stream=stream0)
        del buf739
        buf741 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf740, reinterpret_tensor(buf737, (4, 4, 256), (1024, 1, 4), 0), out=buf741)
        buf742 = empty_strided_cuda((4, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_65.run(buf741, buf742, 1024, 256, grid=grid(1024, 256), stream=stream0)
        del buf741
        # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
        buf743 = extern_kernels.convolution(reinterpret_tensor(buf742, (4, 256, 16, 16), (65536, 256, 16, 1), 0), primals_1566, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf743, (4, 512, 16, 16), (131072, 256, 16, 1))
        buf744 = empty_strided_cuda((4, 1024, 16, 16), (262144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf743, primals_1567, primals_1568, primals_1569, primals_1570, buf721, buf744, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_212], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_1571, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf745, (4, 512, 16, 16), (131072, 256, 16, 1))
        buf746 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_213, input_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_67.run(buf745, primals_1572, primals_1573, primals_1574, primals_1575, buf746, 524288, grid=grid(524288), stream=stream0)
        del primals_1575
        # Topologically Sorted Source Nodes: [out_768], Original ATen: [aten.convolution]
        buf747 = extern_kernels.convolution(buf746, primals_1576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf747, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf748 = buf747; del buf747  # reuse
        # Topologically Sorted Source Nodes: [out_768], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_68.run(buf748, primals_1577, 4096, grid=grid(4096), stream=stream0)
        del primals_1577
    return (buf748, buf718, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_465, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_525, primals_527, primals_528, primals_529, primals_530, primals_532, primals_533, primals_534, primals_535, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_562, primals_563, primals_564, primals_565, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_595, primals_597, primals_598, primals_599, primals_600, primals_602, primals_603, primals_604, primals_605, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_627, primals_628, primals_629, primals_630, primals_632, primals_633, primals_634, primals_635, primals_637, primals_638, primals_639, primals_640, primals_642, primals_643, primals_644, primals_645, primals_647, primals_648, primals_649, primals_650, primals_652, primals_653, primals_654, primals_655, primals_657, primals_658, primals_659, primals_660, primals_662, primals_663, primals_664, primals_665, primals_667, primals_668, primals_669, primals_670, primals_672, primals_673, primals_674, primals_675, primals_677, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_685, primals_687, primals_688, primals_689, primals_690, primals_692, primals_693, primals_694, primals_695, primals_697, primals_698, primals_699, primals_700, primals_702, primals_703, primals_704, primals_705, primals_707, primals_708, primals_709, primals_710, primals_712, primals_713, primals_714, primals_715, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_735, primals_737, primals_738, primals_739, primals_740, primals_742, primals_743, primals_744, primals_745, primals_747, primals_748, primals_749, primals_750, primals_752, primals_753, primals_754, primals_755, primals_757, primals_758, primals_759, primals_760, primals_762, primals_763, primals_764, primals_765, primals_767, primals_768, primals_769, primals_770, primals_772, primals_773, primals_774, primals_775, primals_777, primals_778, primals_779, primals_780, primals_782, primals_783, primals_784, primals_785, primals_787, primals_788, primals_789, primals_790, primals_792, primals_793, primals_794, primals_795, primals_797, primals_798, primals_799, primals_800, primals_802, primals_803, primals_804, primals_805, primals_807, primals_808, primals_809, primals_810, primals_812, primals_813, primals_814, primals_815, primals_817, primals_818, primals_819, primals_820, primals_822, primals_823, primals_824, primals_825, primals_827, primals_828, primals_829, primals_830, primals_832, primals_833, primals_834, primals_835, primals_837, primals_838, primals_839, primals_840, primals_842, primals_843, primals_844, primals_845, primals_847, primals_848, primals_849, primals_850, primals_852, primals_853, primals_854, primals_855, primals_857, primals_858, primals_859, primals_860, primals_862, primals_863, primals_864, primals_865, primals_867, primals_868, primals_869, primals_870, primals_872, primals_873, primals_874, primals_875, primals_877, primals_878, primals_879, primals_880, primals_882, primals_883, primals_884, primals_885, primals_887, primals_888, primals_889, primals_890, primals_892, primals_893, primals_894, primals_895, primals_897, primals_898, primals_899, primals_900, primals_902, primals_903, primals_904, primals_905, primals_907, primals_908, primals_909, primals_910, primals_912, primals_913, primals_914, primals_915, primals_917, primals_918, primals_919, primals_920, primals_922, primals_923, primals_924, primals_925, primals_927, primals_928, primals_929, primals_930, primals_932, primals_933, primals_934, primals_935, primals_937, primals_938, primals_939, primals_940, primals_942, primals_943, primals_944, primals_945, primals_947, primals_948, primals_949, primals_950, primals_952, primals_953, primals_954, primals_955, primals_957, primals_958, primals_959, primals_960, primals_962, primals_963, primals_964, primals_965, primals_967, primals_968, primals_969, primals_970, primals_972, primals_973, primals_974, primals_975, primals_977, primals_978, primals_979, primals_980, primals_982, primals_983, primals_984, primals_985, primals_987, primals_988, primals_989, primals_990, primals_992, primals_993, primals_994, primals_995, primals_997, primals_998, primals_999, primals_1000, primals_1002, primals_1003, primals_1004, primals_1005, primals_1007, primals_1008, primals_1009, primals_1010, primals_1012, primals_1013, primals_1014, primals_1015, primals_1017, primals_1018, primals_1019, primals_1020, primals_1022, primals_1023, primals_1024, primals_1025, primals_1027, primals_1028, primals_1029, primals_1030, primals_1032, primals_1033, primals_1034, primals_1035, primals_1037, primals_1038, primals_1039, primals_1040, primals_1042, primals_1043, primals_1044, primals_1045, primals_1047, primals_1048, primals_1049, primals_1050, primals_1052, primals_1053, primals_1054, primals_1055, primals_1057, primals_1058, primals_1059, primals_1060, primals_1062, primals_1063, primals_1064, primals_1065, primals_1067, primals_1068, primals_1069, primals_1070, primals_1072, primals_1073, primals_1074, primals_1075, primals_1077, primals_1078, primals_1079, primals_1080, primals_1082, primals_1083, primals_1084, primals_1085, primals_1087, primals_1088, primals_1089, primals_1090, primals_1092, primals_1093, primals_1094, primals_1095, primals_1097, primals_1098, primals_1099, primals_1100, primals_1102, primals_1103, primals_1104, primals_1105, primals_1107, primals_1108, primals_1109, primals_1110, primals_1112, primals_1113, primals_1114, primals_1115, primals_1117, primals_1118, primals_1119, primals_1120, primals_1122, primals_1123, primals_1124, primals_1125, primals_1127, primals_1128, primals_1129, primals_1130, primals_1132, primals_1133, primals_1134, primals_1135, primals_1137, primals_1138, primals_1139, primals_1140, primals_1142, primals_1143, primals_1144, primals_1145, primals_1147, primals_1148, primals_1149, primals_1150, primals_1152, primals_1153, primals_1154, primals_1155, primals_1157, primals_1158, primals_1159, primals_1160, primals_1162, primals_1163, primals_1164, primals_1165, primals_1167, primals_1168, primals_1169, primals_1170, primals_1172, primals_1173, primals_1174, primals_1175, primals_1177, primals_1178, primals_1179, primals_1180, primals_1182, primals_1183, primals_1184, primals_1185, primals_1187, primals_1188, primals_1189, primals_1190, primals_1192, primals_1193, primals_1194, primals_1195, primals_1197, primals_1198, primals_1199, primals_1200, primals_1202, primals_1203, primals_1204, primals_1205, primals_1207, primals_1208, primals_1209, primals_1210, primals_1212, primals_1213, primals_1214, primals_1215, primals_1217, primals_1218, primals_1219, primals_1220, primals_1222, primals_1223, primals_1224, primals_1225, primals_1227, primals_1228, primals_1229, primals_1230, primals_1232, primals_1233, primals_1234, primals_1235, primals_1237, primals_1238, primals_1239, primals_1240, primals_1242, primals_1243, primals_1244, primals_1245, primals_1247, primals_1248, primals_1249, primals_1250, primals_1252, primals_1253, primals_1254, primals_1255, primals_1257, primals_1258, primals_1259, primals_1260, primals_1262, primals_1263, primals_1264, primals_1265, primals_1267, primals_1268, primals_1269, primals_1270, primals_1272, primals_1273, primals_1274, primals_1275, primals_1277, primals_1278, primals_1279, primals_1280, primals_1282, primals_1283, primals_1284, primals_1285, primals_1287, primals_1288, primals_1289, primals_1290, primals_1292, primals_1293, primals_1294, primals_1295, primals_1297, primals_1298, primals_1299, primals_1300, primals_1302, primals_1303, primals_1304, primals_1305, primals_1307, primals_1308, primals_1309, primals_1310, primals_1312, primals_1313, primals_1314, primals_1315, primals_1317, primals_1318, primals_1319, primals_1320, primals_1322, primals_1323, primals_1324, primals_1325, primals_1327, primals_1328, primals_1329, primals_1330, primals_1332, primals_1333, primals_1334, primals_1335, primals_1337, primals_1338, primals_1339, primals_1340, primals_1342, primals_1343, primals_1344, primals_1345, primals_1347, primals_1348, primals_1349, primals_1350, primals_1352, primals_1353, primals_1354, primals_1355, primals_1357, primals_1358, primals_1359, primals_1360, primals_1362, primals_1363, primals_1364, primals_1365, primals_1367, primals_1368, primals_1369, primals_1370, primals_1372, primals_1373, primals_1374, primals_1375, primals_1377, primals_1378, primals_1379, primals_1380, primals_1382, primals_1383, primals_1384, primals_1385, primals_1387, primals_1388, primals_1389, primals_1390, primals_1392, primals_1393, primals_1394, primals_1395, primals_1397, primals_1398, primals_1399, primals_1400, primals_1402, primals_1403, primals_1404, primals_1405, primals_1407, primals_1408, primals_1409, primals_1410, primals_1412, primals_1413, primals_1414, primals_1415, primals_1417, primals_1418, primals_1419, primals_1420, primals_1422, primals_1423, primals_1424, primals_1425, primals_1427, primals_1428, primals_1429, primals_1430, primals_1432, primals_1433, primals_1434, primals_1435, primals_1437, primals_1438, primals_1439, primals_1440, primals_1442, primals_1443, primals_1444, primals_1445, primals_1447, primals_1448, primals_1449, primals_1450, primals_1452, primals_1453, primals_1454, primals_1455, primals_1457, primals_1458, primals_1459, primals_1460, primals_1462, primals_1463, primals_1464, primals_1465, primals_1467, primals_1468, primals_1469, primals_1470, primals_1472, primals_1473, primals_1474, primals_1475, primals_1477, primals_1478, primals_1479, primals_1480, primals_1482, primals_1483, primals_1484, primals_1485, primals_1487, primals_1488, primals_1489, primals_1490, primals_1492, primals_1493, primals_1494, primals_1495, primals_1497, primals_1498, primals_1499, primals_1500, primals_1502, primals_1503, primals_1504, primals_1505, primals_1507, primals_1508, primals_1509, primals_1510, primals_1512, primals_1513, primals_1514, primals_1515, primals_1517, primals_1518, primals_1519, primals_1520, primals_1522, primals_1523, primals_1524, primals_1525, primals_1527, primals_1529, primals_1530, primals_1531, primals_1533, primals_1535, primals_1537, primals_1538, primals_1539, primals_1541, primals_1542, primals_1543, primals_1544, primals_1546, primals_1547, primals_1548, primals_1549, primals_1551, primals_1552, primals_1553, primals_1554, primals_1556, primals_1557, primals_1558, primals_1559, primals_1561, primals_1562, primals_1563, primals_1564, primals_1566, primals_1567, primals_1568, primals_1569, primals_1570, primals_1571, primals_1572, primals_1573, primals_1574, primals_1576, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf68, buf69, buf70, buf71, buf72, buf74, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf133, buf135, buf136, buf137, buf138, buf139, buf141, buf143, buf144, buf145, buf147, buf148, buf149, buf150, buf151, buf153, buf155, buf156, buf157, buf158, buf159, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf214, buf218, buf219, buf220, buf224, buf225, buf226, buf227, buf228, buf230, buf231, buf232, buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf246, buf247, buf248, buf249, buf250, buf251, buf252, buf253, buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf273, buf274, buf275, buf276, buf277, buf278, buf279, buf283, buf287, buf288, buf289, buf293, buf294, buf295, buf296, buf297, buf299, buf300, buf301, buf302, buf303, buf304, buf305, buf306, buf307, buf308, buf309, buf310, buf311, buf312, buf313, buf314, buf315, buf316, buf317, buf318, buf319, buf320, buf321, buf322, buf323, buf324, buf325, buf326, buf327, buf328, buf329, buf330, buf331, buf332, buf333, buf334, buf335, buf336, buf337, buf338, buf339, buf340, buf341, buf342, buf343, buf344, buf345, buf346, buf347, buf348, buf352, buf356, buf357, buf358, buf362, buf363, buf364, buf365, buf366, buf368, buf369, buf370, buf371, buf372, buf373, buf374, buf375, buf376, buf377, buf378, buf379, buf380, buf381, buf382, buf383, buf384, buf385, buf386, buf387, buf388, buf389, buf390, buf391, buf392, buf393, buf394, buf395, buf396, buf397, buf398, buf399, buf400, buf401, buf402, buf403, buf404, buf405, buf406, buf407, buf408, buf409, buf410, buf411, buf412, buf413, buf414, buf415, buf416, buf417, buf418, buf419, buf420, buf421, buf422, buf423, buf424, buf425, buf426, buf427, buf428, buf429, buf430, buf431, buf432, buf433, buf434, buf435, buf439, buf443, buf445, buf446, buf447, buf448, buf449, buf451, buf453, buf454, buf455, buf459, buf461, buf462, buf463, buf464, buf465, buf467, buf469, buf470, buf471, buf472, buf473, buf475, buf477, buf478, buf479, buf480, buf481, buf483, buf485, buf486, buf487, buf488, buf489, buf490, buf491, buf492, buf493, buf495, buf496, buf497, buf498, buf499, buf500, buf501, buf502, buf503, buf504, buf505, buf506, buf507, buf508, buf509, buf510, buf511, buf512, buf513, buf514, buf515, buf516, buf517, buf518, buf519, buf520, buf521, buf522, buf523, buf524, buf525, buf526, buf527, buf528, buf529, buf530, buf531, buf532, buf533, buf534, buf535, buf536, buf537, buf538, buf539, buf540, buf541, buf542, buf543, buf544, buf545, buf546, buf547, buf548, buf549, buf550, buf551, buf552, buf553, buf554, buf555, buf556, buf557, buf558, buf559, buf560, buf561, buf565, buf569, buf573, buf574, buf575, buf579, buf583, buf584, buf585, buf586, buf587, buf589, buf593, buf594, buf595, buf596, buf597, buf598, buf599, buf600, buf601, buf603, buf604, buf605, buf606, buf607, buf608, buf609, buf610, buf611, buf612, buf613, buf614, buf615, buf616, buf617, buf618, buf619, buf620, buf621, buf622, buf623, buf624, buf625, buf626, buf627, buf628, buf629, buf630, buf631, buf632, buf633, buf634, buf635, buf636, buf637, buf638, buf639, buf640, buf641, buf642, buf643, buf644, buf645, buf646, buf647, buf648, buf649, buf650, buf651, buf652, buf653, buf654, buf655, buf656, buf657, buf658, buf659, buf660, buf661, buf662, buf663, buf664, buf665, buf666, buf667, buf668, buf669, buf673, buf677, buf681, buf682, buf686, buf690, buf691, buf692, buf693, buf695, buf699, buf700, buf701, buf702, buf703, buf704, buf705, buf706, buf708, buf713, buf715, buf716, buf720, buf721, buf724, reinterpret_tensor(buf725, (4, 512, 4, 1), (2048, 1, 512, 1), 0), buf726, buf727, buf728, buf731, buf732, buf733, buf736, buf740, reinterpret_tensor(buf742, (4, 256, 16, 16), (65536, 256, 16, 1), 0), buf743, buf744, buf745, buf746, reinterpret_tensor(buf737, (4, 256, 4), (1024, 4, 1), 0), reinterpret_tensor(buf729, (4, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf734, (4, 4, 256), (1024, 1, 4), 0), buf749, buf750, buf751, buf752, buf753, buf754, buf755, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((8, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1104 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1105 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1106 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1107 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1108 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1109 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1110 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1111 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1112 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1113 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1114 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1115 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1116 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1117 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1118 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1119 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1120 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1121 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1122 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1123 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1124 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1125 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1126 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1127 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1128 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1129 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1130 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1131 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1132 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1133 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1134 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1135 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1136 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1137 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1138 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1139 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1140 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1141 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1142 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1143 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1144 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1145 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1146 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1147 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1148 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1149 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1150 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1151 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1152 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1153 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1154 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1155 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1156 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1157 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1158 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1159 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1160 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1161 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1162 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1163 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1164 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1167 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1168 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1170 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1172 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1173 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1176 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1177 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1178 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1180 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1181 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1182 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1187 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1192 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1197 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1202 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1207 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1208 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1209 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1210 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1211 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1212 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1213 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1214 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1215 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1216 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1217 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1218 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1219 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1220 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1221 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1222 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1223 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1224 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1225 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1226 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1227 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1228 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1229 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1230 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1231 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1232 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1233 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1234 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1235 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1236 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1237 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1238 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1239 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1240 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1241 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1242 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1243 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1244 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1245 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1246 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1247 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1248 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1249 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1250 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1251 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1252 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1253 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1254 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1255 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1256 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1257 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1258 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1259 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1260 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1261 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1262 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1263 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1264 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1265 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1266 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1267 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1269 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1270 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1272 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1273 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1274 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1275 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1276 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1277 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1278 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1279 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1280 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1281 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1282 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1283 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1284 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1285 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1287 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1288 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1289 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1290 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1291 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1292 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1293 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1294 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1295 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1296 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1297 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1298 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1299 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1300 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1301 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1302 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1303 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1304 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1305 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1306 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1307 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1308 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1309 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1310 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1311 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1312 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1313 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1314 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1315 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1316 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1317 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1318 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1319 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1320 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1321 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1322 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1323 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1324 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1325 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1326 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1327 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1328 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1329 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1330 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1331 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1332 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1333 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1334 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1335 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1336 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1337 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1338 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1339 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1340 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1341 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1342 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1343 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1344 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1345 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1346 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1347 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1348 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1349 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1350 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1351 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1352 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1353 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1354 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1355 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1356 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1357 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1358 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1359 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1360 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1361 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1362 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1363 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1364 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1365 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1366 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1367 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1368 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1369 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1370 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1371 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1372 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1373 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1374 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1375 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1376 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1377 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1378 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1379 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1380 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1381 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1382 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1383 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1384 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1385 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1386 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1387 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1388 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1389 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1390 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1391 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1392 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1393 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1394 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1395 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1396 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1397 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1398 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1399 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1400 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1401 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1402 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1403 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1404 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1405 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1406 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1407 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1408 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1409 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1410 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1411 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1412 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1413 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1414 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1415 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1416 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1417 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1418 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1419 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1420 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1421 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1422 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1423 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1424 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1425 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1426 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1427 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1428 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1429 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1430 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1431 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1432 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1433 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1434 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1435 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1436 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1437 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1438 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1439 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1440 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1441 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1442 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1443 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1444 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1445 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1446 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1447 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1448 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1449 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1450 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1451 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1452 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1453 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1454 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1455 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1456 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1457 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1458 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1459 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1460 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1461 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1462 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1463 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1464 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1465 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1466 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1467 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1468 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1469 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1470 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1471 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1472 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1473 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1474 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1475 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1476 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1477 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1478 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1479 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1480 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1481 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1482 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1483 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1484 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1485 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1486 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1487 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1488 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1489 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1490 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1491 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1492 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1493 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1494 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1495 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1496 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1497 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1498 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1499 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1500 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1501 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1502 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1503 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1504 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1505 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1506 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1507 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1508 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1509 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1510 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1511 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1512 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1513 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1514 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1515 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1516 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1517 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1518 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1519 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1520 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1521 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1522 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1523 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1524 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1525 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1526 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1527 = rand_strided((60, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1528 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1529 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1530 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1531 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1532 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1533 = rand_strided((4, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1534 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1535 = rand_strided((512, 60, 3, 3), (540, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1536 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1537 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1538 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1539 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1540 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1541 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1542 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1543 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1544 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1545 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1546 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1547 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1548 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1549 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1550 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1551 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1552 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1553 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1554 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1555 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1556 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1557 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1558 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1559 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1560 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1561 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1562 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1563 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1564 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1565 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1566 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1567 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1568 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1569 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1570 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1571 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1572 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1573 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1574 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1575 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1576 = rand_strided((4, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1577 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463, primals_1464, primals_1465, primals_1466, primals_1467, primals_1468, primals_1469, primals_1470, primals_1471, primals_1472, primals_1473, primals_1474, primals_1475, primals_1476, primals_1477, primals_1478, primals_1479, primals_1480, primals_1481, primals_1482, primals_1483, primals_1484, primals_1485, primals_1486, primals_1487, primals_1488, primals_1489, primals_1490, primals_1491, primals_1492, primals_1493, primals_1494, primals_1495, primals_1496, primals_1497, primals_1498, primals_1499, primals_1500, primals_1501, primals_1502, primals_1503, primals_1504, primals_1505, primals_1506, primals_1507, primals_1508, primals_1509, primals_1510, primals_1511, primals_1512, primals_1513, primals_1514, primals_1515, primals_1516, primals_1517, primals_1518, primals_1519, primals_1520, primals_1521, primals_1522, primals_1523, primals_1524, primals_1525, primals_1526, primals_1527, primals_1528, primals_1529, primals_1530, primals_1531, primals_1532, primals_1533, primals_1534, primals_1535, primals_1536, primals_1537, primals_1538, primals_1539, primals_1540, primals_1541, primals_1542, primals_1543, primals_1544, primals_1545, primals_1546, primals_1547, primals_1548, primals_1549, primals_1550, primals_1551, primals_1552, primals_1553, primals_1554, primals_1555, primals_1556, primals_1557, primals_1558, primals_1559, primals_1560, primals_1561, primals_1562, primals_1563, primals_1564, primals_1565, primals_1566, primals_1567, primals_1568, primals_1569, primals_1570, primals_1571, primals_1572, primals_1573, primals_1574, primals_1575, primals_1576, primals_1577])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
