# AOT ID: ['18_forward']
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


# kernel path: inductor_cache/xs/cxsckpgvhc5gwksvb6t3nymnmfkjluxma5ztdxtrwcmi25tnxr72.py
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
    x1 = ((xindex // 256) % 32)
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


# kernel path: inductor_cache/ci/ccio6blyt64aprmkdlnuy7oahxas27vzqkussnn3sh42bmbciclh.py
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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


# kernel path: inductor_cache/nj/cnjuxmqnicktlz45bqddy5iknkftgoqkf2oupikwncjtw735oqmb.py
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
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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


# kernel path: inductor_cache/ad/cad4klwvg244ets5qzfoz57f6r2ejxp2vyzm5smuxta4nqe4qjac.py
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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


# kernel path: inductor_cache/tr/ctrm4yk77zdxw6cxrzltyqngd4ovnf5sroelmrit24zfr6shqe52.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_11 => add_80, add_81, convert_element_type_68, convert_element_type_69, iota, mul_102, mul_103
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, 0), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_80, torch.float32), kwargs = {})
#   %add_81 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, 0.0), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, 0.5), kwargs = {})
#   %convert_element_type_69 : [num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_103, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_8 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cchdys23xbjj4byttsxlbbshet63trnwvn5qbrdzb76nzdnmqck7.py
# Topologically Sorted Source Nodes: [out_65, out_66, out_67, input_10, input_11, value, value_1, xi], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_10 => add_79, mul_100, mul_101, sub_33
#   input_11 => _unsafe_index
#   out_65 => add_56, mul_73, mul_74, sub_24
#   out_66 => add_57
#   out_67 => relu_23
#   value => add_84
#   value_1 => add_85
#   xi => relu_32
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %relu_21), kwargs = {})
#   %relu_23 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_79, [None, None, %unsqueeze_272, %convert_element_type_69]), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_23, 0), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %_unsafe_index), kwargs = {})
#   %relu_32 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 32)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x5), None)
    tmp22 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp23 = tl.full([XBLOCK], 8, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tmp28 = tmp27 + tmp23
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr7 + (tmp30 + 8*tmp26 + 64*x6), None, eviction_policy='evict_last')
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 + tmp4
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tmp7 / tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp21 + tmp43
    tmp45 = triton_helpers.maximum(tmp18, tmp44)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(out_ptr1 + (x5), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/n2/cn2f5yclasqaljrkw56k5ofunqbqsxvuoh2lbrhrhesghe77uclk.py
# Topologically Sorted Source Nodes: [input_13, value_2, value_3, xi_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_13 => add_87, mul_107, mul_108, sub_34
#   value_2 => add_88
#   value_3 => add_89
#   xi_1 => relu_33
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_274), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_276), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %unsqueeze_278), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %unsqueeze_280), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, 0), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %relu_31), kwargs = {})
#   %relu_33 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_89,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmzs64iw3i7yh2nzi7ihqr75vvr6calr4vdg76akxdhigaw4q5fp.py
# Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_15 => add_91, mul_110, mul_111, sub_35
#   input_16 => relu_34
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_282), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_284), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %unsqueeze_286), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %unsqueeze_288), kwargs = {})
#   %relu_34 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ic/ciczind6qdbrlptkbryrse45hfxhs73v4ilf7e7rwujaeizchz63.py
# Topologically Sorted Source Nodes: [out_156, out_157, out_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_156 => add_135, mul_164, mul_165, sub_53
#   out_157 => add_136
#   out_158 => relu_52
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_426), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_428), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, %unsqueeze_430), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_165, %unsqueeze_432), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %relu_34), kwargs = {})
#   %relu_52 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_136,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/aj/cajvvyhmvsx5mp7zcrnc6uwimogblyvve4stu6eidulztw4axcv5.py
# Topologically Sorted Source Nodes: [input_11, input_22], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_11 => add_80, add_81, convert_element_type_68, iota, mul_102
#   input_22 => convert_element_type_133, mul_195
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, 0), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_80, torch.float32), kwargs = {})
#   %add_81 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, 0.0), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, 0.25), kwargs = {})
#   %convert_element_type_133 : [num_users=9] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_195, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_13 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2e/c2eskzzosqbr5n5uvetjnf4s2mtcehoe7n33pvluofwxijddpnav.py
# Topologically Sorted Source Nodes: [out_121, out_122, out_123, input_18, input_19, input_21, input_22, value_4, value_5, value_6, xi_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_18 => add_153, mul_185, mul_186, sub_60
#   input_19 => _unsafe_index_1
#   input_21 => add_159, mul_192, mul_193, sub_61
#   input_22 => _unsafe_index_2
#   out_121 => add_110, mul_134, mul_135, sub_43
#   out_122 => add_111
#   out_123 => relu_42
#   value_4 => add_164
#   value_5 => add_165
#   value_6 => add_166
#   xi_2 => relu_59
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_346), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_348), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %unsqueeze_350), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %unsqueeze_352), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, %relu_40), kwargs = {})
#   %relu_42 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_482), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_484), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_185, %unsqueeze_486), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, %unsqueeze_488), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_153, [None, None, %unsqueeze_272, %convert_element_type_69]), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_491), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_493), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_495), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_497), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_159, [None, None, %unsqueeze_498, %convert_element_type_133]), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_42, 0), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %_unsafe_index_1), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %_unsafe_index_2), kwargs = {})
#   %relu_59 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_166,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 32)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x5), None)
    tmp22 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x3), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp23 = tl.full([XBLOCK], 8, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tmp28 = tmp27 + tmp23
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr7 + (tmp30 + 8*tmp26 + 64*x6), None, eviction_policy='evict_last')
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 + tmp4
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tmp7 / tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp21 + tmp43
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tmp51 = tmp50 + tmp46
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr13 + (tmp53 + 4*tmp49 + 16*x6), None, eviction_policy='evict_last')
    tmp56 = tmp54 - tmp55
    tmp58 = tmp57 + tmp4
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tmp7 / tmp59
    tmp61 = tmp60 * tmp9
    tmp62 = tmp56 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp44 + tmp66
    tmp68 = triton_helpers.maximum(tmp18, tmp67)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(in_out_ptr0 + (x5), tmp68, None)
''', device_str='cuda')


# kernel path: inductor_cache/gw/cgwbfofjt2kysxmidfb2fuxhstwxsetlvgsl4p3hn3lzbd7xzt7m.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_27 => add_171, add_172, convert_element_type_140, convert_element_type_141, iota_6, mul_204, mul_205
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_204, 0), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_171, torch.float32), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_140, 0.0), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_172, 0.5), kwargs = {})
#   %convert_element_type_141 : [num_users=8] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_205, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_15 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7k/c7klu7fbxvmlvth3uetc6y32a3m2hfj7vbm2dvdsg6yvrfzxand7.py
# Topologically Sorted Source Nodes: [input_24, input_26, input_27, value_7, value_8, value_9, xi_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_24 => add_168, mul_199, mul_200, sub_62
#   input_26 => add_170, mul_202, mul_203, sub_63
#   input_27 => _unsafe_index_3
#   value_7 => add_175
#   value_8 => add_176
#   value_9 => add_177
#   xi_3 => relu_60
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_500), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_502), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_504), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_506), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_508), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_510), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_512), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_514), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_170, [None, None, %unsqueeze_515, %convert_element_type_141]), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, 0), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_175, %relu_50), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %_unsafe_index_3), kwargs = {})
#   %relu_60 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x2 = ((xindex // 64) % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x6 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x4), None)
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tl.full([XBLOCK], 4, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr7 + (tmp28 + 4*tmp24 + 16*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp4
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp7 / tmp34
    tmp36 = tmp35 * tmp9
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tmp43 = tl.full([1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(in_out_ptr0 + (x4), tmp44, None)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdba7hpznlvavv3wcpxirqmmahvrjh6kfhnncg5tnr4b76lq4em.py
# Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_29 => add_179, mul_209, mul_210, sub_64
#   input_30 => relu_61
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_517), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_519), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_209, %unsqueeze_521), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_210, %unsqueeze_523), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_179,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xe/cxerg4dnpxlwto2b2nsvsncia77hj4xjf6g6czp2bmbt7tz34i4u.py
# Topologically Sorted Source Nodes: [input_32, input_34, value_10, value_11, value_12, xi_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_32 => add_181, mul_212, mul_213, sub_65
#   input_34 => add_183, mul_215, mul_216, sub_66
#   value_10 => add_184
#   value_11 => add_185
#   value_12 => add_186
#   xi_4 => relu_62
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_525), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_527), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_212, %unsqueeze_529), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_213, %unsqueeze_531), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_533), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_535), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_215, %unsqueeze_537), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_216, %unsqueeze_539), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_181, 0), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %add_183), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_185, %relu_58), kwargs = {})
#   %relu_62 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_186,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x3), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/56/c564yowtpd3rxg22ve4qyjbpig5atawsosflq3hpqqcc3j6v5ren.py
# Topologically Sorted Source Nodes: [input_90, input_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_90 => add_473, mul_533, mul_534, sub_160
#   input_91 => relu_147
# Graph fragment:
#   %sub_160 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_1294), kwargs = {})
#   %mul_533 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_160, %unsqueeze_1296), kwargs = {})
#   %mul_534 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_533, %unsqueeze_1298), kwargs = {})
#   %add_473 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_534, %unsqueeze_1300), kwargs = {})
#   %relu_147 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ln/clnnq7qfdewf6wqwz2pj5xe2mqa5q7rt7mzeyke5gmapzrgmah43.py
# Topologically Sorted Source Nodes: [out_520, out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_520 => add_537, mul_611, mul_612, sub_186
#   out_521 => add_538
#   out_522 => relu_173
# Graph fragment:
#   %sub_186 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_186, %unsqueeze_1502), kwargs = {})
#   %mul_611 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_186, %unsqueeze_1504), kwargs = {})
#   %mul_612 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_611, %unsqueeze_1506), kwargs = {})
#   %add_537 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_612, %unsqueeze_1508), kwargs = {})
#   %add_538 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_537, %relu_147), kwargs = {})
#   %relu_173 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_538,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/kq/ckqkj3ryzebfstgba3i2rffzagi26eagyo6krtdnqgbjokp5mgzo.py
# Topologically Sorted Source Nodes: [input_11, input_100], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_100 => convert_element_type_453, mul_649
#   input_11 => add_80, add_81, convert_element_type_68, iota, mul_102
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, 0), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_80, torch.float32), kwargs = {})
#   %add_81 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, 0.0), kwargs = {})
#   %mul_649 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, 0.125), kwargs = {})
#   %convert_element_type_453 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_649, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_21 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.125
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rc/crcpal4eqpjlynz2ar2foqzrt3opf6gffylnlpcx7sisplemmjxb.py
# Topologically Sorted Source Nodes: [out_457, out_458, out_459, input_93, input_94, input_96, input_97, input_99, input_100, value_40, value_41, value_42, value_43, xi_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_100 => _unsafe_index_15
#   input_93 => add_555, mul_632, mul_633, sub_193
#   input_94 => _unsafe_index_13
#   input_96 => add_561, mul_639, mul_640, sub_194
#   input_97 => _unsafe_index_14
#   input_99 => add_567, mul_646, mul_647, sub_195
#   out_457 => add_492, mul_557, mul_558, sub_168
#   out_458 => add_493
#   out_459 => relu_155
#   value_40 => add_572
#   value_41 => add_573
#   value_42 => add_574
#   value_43 => add_575
#   xi_14 => relu_180
# Graph fragment:
#   %sub_168 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_168, %unsqueeze_1358), kwargs = {})
#   %mul_557 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_168, %unsqueeze_1360), kwargs = {})
#   %mul_558 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_557, %unsqueeze_1362), kwargs = {})
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_558, %unsqueeze_1364), kwargs = {})
#   %add_493 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_492, %relu_153), kwargs = {})
#   %relu_155 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%add_493,), kwargs = {})
#   %sub_193 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_193, %unsqueeze_1558), kwargs = {})
#   %mul_632 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_193, %unsqueeze_1560), kwargs = {})
#   %mul_633 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_632, %unsqueeze_1562), kwargs = {})
#   %add_555 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_633, %unsqueeze_1564), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_555, [None, None, %unsqueeze_272, %convert_element_type_69]), kwargs = {})
#   %sub_194 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_194, %unsqueeze_1567), kwargs = {})
#   %mul_639 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_194, %unsqueeze_1569), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_639, %unsqueeze_1571), kwargs = {})
#   %add_561 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_640, %unsqueeze_1573), kwargs = {})
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_561, [None, None, %unsqueeze_498, %convert_element_type_133]), kwargs = {})
#   %sub_195 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_195, %unsqueeze_1576), kwargs = {})
#   %mul_646 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_195, %unsqueeze_1578), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_646, %unsqueeze_1580), kwargs = {})
#   %add_567 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_647, %unsqueeze_1582), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_567, [None, None, %unsqueeze_1583, %convert_element_type_453]), kwargs = {})
#   %add_572 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_155, 0), kwargs = {})
#   %add_573 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_572, %_unsafe_index_13), kwargs = {})
#   %add_574 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_573, %_unsafe_index_14), kwargs = {})
#   %add_575 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_574, %_unsafe_index_15), kwargs = {})
#   %relu_180 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_575,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 32)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x5), None)
    tmp22 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x3), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr18 + (x3), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr20 + (x1), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr21 + (x1), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr22 + (x1), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr23 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp23 = tl.full([XBLOCK], 8, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tmp28 = tmp27 + tmp23
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr7 + (tmp30 + 8*tmp26 + 64*x6), None, eviction_policy='evict_last')
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 + tmp4
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tmp7 / tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp21 + tmp43
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tmp51 = tmp50 + tmp46
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr13 + (tmp53 + 4*tmp49 + 16*x6), None, eviction_policy='evict_last')
    tmp56 = tmp54 - tmp55
    tmp58 = tmp57 + tmp4
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tmp7 / tmp59
    tmp61 = tmp60 * tmp9
    tmp62 = tmp56 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp44 + tmp66
    tmp69 = tl.full([XBLOCK], 2, tl.int32)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp68 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp68)
    tmp74 = tmp73 + tmp69
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tmp77 = tl.load(in_ptr19 + (tmp76 + 2*tmp72 + 4*x6), None, eviction_policy='evict_last')
    tmp79 = tmp77 - tmp78
    tmp81 = tmp80 + tmp4
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tmp7 / tmp82
    tmp84 = tmp83 * tmp9
    tmp85 = tmp79 * tmp84
    tmp87 = tmp85 * tmp86
    tmp89 = tmp87 + tmp88
    tmp90 = tmp67 + tmp89
    tmp91 = triton_helpers.maximum(tmp18, tmp90)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(in_out_ptr0 + (x5), tmp91, None)
''', device_str='cuda')


# kernel path: inductor_cache/qg/cqgjil3ld4odm3cghsdz5mqngxne5fwntaddjhbdy5ikxyqirk7a.py
# Topologically Sorted Source Nodes: [input_27, input_108], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_108 => convert_element_type_467, mul_666
#   input_27 => add_171, add_172, convert_element_type_140, iota_6, mul_204
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_204, 0), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_171, torch.float32), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_140, 0.0), kwargs = {})
#   %mul_666 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_172, 0.25), kwargs = {})
#   %convert_element_type_467 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_666, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_23 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cz/ccz7onib4oouj4wqvtz5z5qgueepoo5nklc32czomvyuxwtsovx6.py
# Topologically Sorted Source Nodes: [input_102, input_104, input_105, input_107, input_108, value_44, value_45, value_46, value_47, xi_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_102 => add_577, mul_653, mul_654, sub_196
#   input_104 => add_579, mul_656, mul_657, sub_197
#   input_105 => _unsafe_index_16
#   input_107 => add_585, mul_663, mul_664, sub_198
#   input_108 => _unsafe_index_17
#   value_44 => add_590
#   value_45 => add_591
#   value_46 => add_592
#   value_47 => add_593
#   xi_15 => relu_181
# Graph fragment:
#   %sub_196 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_196, %unsqueeze_1585), kwargs = {})
#   %mul_653 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_196, %unsqueeze_1587), kwargs = {})
#   %mul_654 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_653, %unsqueeze_1589), kwargs = {})
#   %add_577 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_654, %unsqueeze_1591), kwargs = {})
#   %sub_197 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_197, %unsqueeze_1593), kwargs = {})
#   %mul_656 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_197, %unsqueeze_1595), kwargs = {})
#   %mul_657 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_656, %unsqueeze_1597), kwargs = {})
#   %add_579 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_657, %unsqueeze_1599), kwargs = {})
#   %_unsafe_index_16 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_579, [None, None, %unsqueeze_515, %convert_element_type_141]), kwargs = {})
#   %sub_198 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_1602), kwargs = {})
#   %mul_663 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_198, %unsqueeze_1604), kwargs = {})
#   %mul_664 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_663, %unsqueeze_1606), kwargs = {})
#   %add_585 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_664, %unsqueeze_1608), kwargs = {})
#   %_unsafe_index_17 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_585, [None, None, %unsqueeze_1609, %convert_element_type_467]), kwargs = {})
#   %add_590 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_577, 0), kwargs = {})
#   %add_591 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_590, %relu_163), kwargs = {})
#   %add_592 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_591, %_unsafe_index_16), kwargs = {})
#   %add_593 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_592, %_unsafe_index_17), kwargs = {})
#   %relu_181 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_593,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x2 = ((xindex // 64) % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x6 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x4), None)
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr14 + (x2), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr15 + (x2), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr16 + (x2), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr17 + (x2), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tl.full([XBLOCK], 4, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr7 + (tmp28 + 4*tmp24 + 16*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp4
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp7 / tmp34
    tmp36 = tmp35 * tmp9
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tmp44 = tl.full([XBLOCK], 2, tl.int32)
    tmp45 = tmp43 + tmp44
    tmp46 = tmp43 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp43)
    tmp49 = tmp48 + tmp44
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tmp52 = tl.load(in_ptr13 + (tmp51 + 2*tmp47 + 4*x6), None, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp4
    tmp57 = libdevice.sqrt(tmp56)
    tmp58 = tmp7 / tmp57
    tmp59 = tmp58 * tmp9
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp42 + tmp64
    tmp66 = tl.full([1], 0, tl.int32)
    tmp67 = triton_helpers.maximum(tmp66, tmp65)
    tl.store(in_out_ptr0 + (x4), tmp67, None)
''', device_str='cuda')


# kernel path: inductor_cache/u2/cu23yplsvtczbhjpsemylbwwsmxycsceeb5kev6y2grbp7pza3px.py
# Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_118 => add_602, add_603, convert_element_type_478, convert_element_type_479, iota_36, mul_681, mul_682
# Graph fragment:
#   %iota_36 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_681 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_36, 1), kwargs = {})
#   %add_602 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_681, 0), kwargs = {})
#   %convert_element_type_478 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_602, torch.float32), kwargs = {})
#   %add_603 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_478, 0.0), kwargs = {})
#   %mul_682 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_603, 0.5), kwargs = {})
#   %convert_element_type_479 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_682, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_25 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujfgqxydiffjxesrqnot74gebql4xx5uf7v3zpn75gcevwu443p.py
# Topologically Sorted Source Nodes: [input_113, input_115, input_117, input_118, value_48, value_49, value_50, value_51, xi_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_113 => add_597, mul_673, mul_674, sub_200
#   input_115 => add_599, mul_676, mul_677, sub_201
#   input_117 => add_601, mul_679, mul_680, sub_202
#   input_118 => _unsafe_index_18
#   value_48 => add_606
#   value_49 => add_607
#   value_50 => add_608
#   value_51 => add_609
#   xi_16 => relu_183
# Graph fragment:
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_200, %unsqueeze_1619), kwargs = {})
#   %mul_673 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_200, %unsqueeze_1621), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_673, %unsqueeze_1623), kwargs = {})
#   %add_597 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_674, %unsqueeze_1625), kwargs = {})
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1627), kwargs = {})
#   %mul_676 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1629), kwargs = {})
#   %mul_677 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_676, %unsqueeze_1631), kwargs = {})
#   %add_599 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_677, %unsqueeze_1633), kwargs = {})
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_1635), kwargs = {})
#   %mul_679 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1637), kwargs = {})
#   %mul_680 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_679, %unsqueeze_1639), kwargs = {})
#   %add_601 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_680, %unsqueeze_1641), kwargs = {})
#   %_unsafe_index_18 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_601, [None, None, %unsqueeze_1642, %convert_element_type_479]), kwargs = {})
#   %add_606 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_597, 0), kwargs = {})
#   %add_607 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_606, %add_599), kwargs = {})
#   %add_608 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_607, %relu_171), kwargs = {})
#   %add_609 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_608, %_unsafe_index_18), kwargs = {})
#   %relu_183 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_609,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 16) % 128)
    x4 = ((xindex // 4) % 4)
    x3 = (xindex % 4)
    x6 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x5), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x5), None)
    tmp34 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x3), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tl.full([XBLOCK], 2, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tmp40 = tmp39 + tmp35
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr12 + (tmp42 + 2*tmp38 + 4*x6), None, eviction_policy='evict_last')
    tmp45 = tmp43 - tmp44
    tmp47 = tmp46 + tmp4
    tmp48 = libdevice.sqrt(tmp47)
    tmp49 = tmp7 / tmp48
    tmp50 = tmp49 * tmp9
    tmp51 = tmp45 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tmp33 + tmp55
    tmp57 = tl.full([1], 0, tl.int32)
    tmp58 = triton_helpers.maximum(tmp57, tmp56)
    tl.store(in_out_ptr0 + (x5), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7n5ywkc7iwep4b7py5ixoqdhhl2ltwhmuzpieb7bmtp5tznvalm.py
# Topologically Sorted Source Nodes: [input_123, input_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_123 => add_613, mul_689, mul_690, sub_204
#   input_124 => relu_185
# Graph fragment:
#   %sub_204 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1652), kwargs = {})
#   %mul_689 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_204, %unsqueeze_1654), kwargs = {})
#   %mul_690 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_689, %unsqueeze_1656), kwargs = {})
#   %add_613 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_690, %unsqueeze_1658), kwargs = {})
#   %relu_185 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_613,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/w6/cw6jc6u7btz4wnjq33ittmsxbochaop5vv2iwgqwvragnaoecoga.py
# Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_128 => add_617, mul_695, mul_696, sub_206
#   input_129 => relu_186
# Graph fragment:
#   %sub_206 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_206, %unsqueeze_1668), kwargs = {})
#   %mul_695 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_206, %unsqueeze_1670), kwargs = {})
#   %mul_696 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_695, %unsqueeze_1672), kwargs = {})
#   %add_617 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_696, %unsqueeze_1674), kwargs = {})
#   %relu_186 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_617,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
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


# kernel path: inductor_cache/q3/cq3ojo5gt7qgnsgmkxw2cbhnbo5i7x3nqbkjpjpk5uxqtz5szf43.py
# Topologically Sorted Source Nodes: [input_126, input_131, input_133, value_52, value_53, value_54, value_55, xi_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_126 => add_615, mul_692, mul_693, sub_205
#   input_131 => add_619, mul_698, mul_699, sub_207
#   input_133 => add_621, mul_701, mul_702, sub_208
#   value_52 => add_622
#   value_53 => add_623
#   value_54 => add_624
#   value_55 => add_625
#   xi_17 => relu_187
# Graph fragment:
#   %sub_205 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_205, %unsqueeze_1660), kwargs = {})
#   %mul_692 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %unsqueeze_1662), kwargs = {})
#   %mul_693 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_692, %unsqueeze_1664), kwargs = {})
#   %add_615 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_693, %unsqueeze_1666), kwargs = {})
#   %sub_207 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_207, %unsqueeze_1676), kwargs = {})
#   %mul_698 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_207, %unsqueeze_1678), kwargs = {})
#   %mul_699 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_698, %unsqueeze_1680), kwargs = {})
#   %add_619 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_699, %unsqueeze_1682), kwargs = {})
#   %sub_208 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_208, %unsqueeze_1684), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_208, %unsqueeze_1686), kwargs = {})
#   %mul_702 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_701, %unsqueeze_1688), kwargs = {})
#   %add_621 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_702, %unsqueeze_1690), kwargs = {})
#   %add_622 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_615, 0), kwargs = {})
#   %add_623 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_622, %add_619), kwargs = {})
#   %add_624 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_623, %add_621), kwargs = {})
#   %add_625 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_624, %relu_179), kwargs = {})
#   %relu_187 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_625,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x3), None)
    tmp33 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr15 + (x3), None)
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
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp34 = tmp32 - tmp33
    tmp36 = tmp35 + tmp4
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tmp7 / tmp37
    tmp39 = tmp38 * tmp9
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tmp31 + tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tl.store(in_out_ptr0 + (x3), tmp49, None)
''', device_str='cuda')


# kernel path: inductor_cache/fb/cfbqxi2d2aenhf7mu3l4r4xjlhcrlihj4vgot3m56zx7dodzbbgs.py
# Topologically Sorted Source Nodes: [out_681, out_682, out_683, input_177, input_178, input_180, input_181, input_183, input_184, value_72, value_73, value_74, value_75, relu_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_177 => add_859, mul_968, mul_969, sub_289
#   input_178 => _unsafe_index_25
#   input_180 => add_865, mul_975, mul_976, sub_290
#   input_181 => _unsafe_index_26
#   input_183 => add_871, mul_982, mul_983, sub_291
#   input_184 => _unsafe_index_27
#   out_681 => add_796, mul_893, mul_894, sub_264
#   out_682 => add_797
#   out_683 => relu_235
#   relu_260 => relu_260
#   value_72 => add_876
#   value_73 => add_877
#   value_74 => add_878
#   value_75 => add_879
# Graph fragment:
#   %sub_264 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_264, %unsqueeze_2138), kwargs = {})
#   %mul_893 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_264, %unsqueeze_2140), kwargs = {})
#   %mul_894 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_893, %unsqueeze_2142), kwargs = {})
#   %add_796 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_894, %unsqueeze_2144), kwargs = {})
#   %add_797 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_796, %relu_233), kwargs = {})
#   %relu_235 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_797,), kwargs = {})
#   %sub_289 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_289, %unsqueeze_2338), kwargs = {})
#   %mul_968 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_289, %unsqueeze_2340), kwargs = {})
#   %mul_969 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_968, %unsqueeze_2342), kwargs = {})
#   %add_859 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_969, %unsqueeze_2344), kwargs = {})
#   %_unsafe_index_25 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_859, [None, None, %unsqueeze_272, %convert_element_type_69]), kwargs = {})
#   %sub_290 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_290, %unsqueeze_2347), kwargs = {})
#   %mul_975 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_290, %unsqueeze_2349), kwargs = {})
#   %mul_976 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_975, %unsqueeze_2351), kwargs = {})
#   %add_865 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_976, %unsqueeze_2353), kwargs = {})
#   %_unsafe_index_26 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_865, [None, None, %unsqueeze_498, %convert_element_type_133]), kwargs = {})
#   %sub_291 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_291, %unsqueeze_2356), kwargs = {})
#   %mul_982 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_291, %unsqueeze_2358), kwargs = {})
#   %mul_983 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_982, %unsqueeze_2360), kwargs = {})
#   %add_871 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_983, %unsqueeze_2362), kwargs = {})
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_871, [None, None, %unsqueeze_1583, %convert_element_type_453]), kwargs = {})
#   %add_876 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_235, 0), kwargs = {})
#   %add_877 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_876, %_unsafe_index_25), kwargs = {})
#   %add_878 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_877, %_unsafe_index_26), kwargs = {})
#   %add_879 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_878, %_unsafe_index_27), kwargs = {})
#   %relu_260 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_879,), kwargs = {})
#   %le_25 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_235, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_threshold_backward_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_threshold_backward_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_threshold_backward_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_threshold_backward_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 32)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x5), None)
    tmp22 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x3), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr18 + (x3), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr20 + (x1), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr21 + (x1), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr22 + (x1), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr23 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp23 = tl.full([XBLOCK], 8, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tmp28 = tmp27 + tmp23
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr7 + (tmp30 + 8*tmp26 + 64*x6), None, eviction_policy='evict_last')
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 + tmp4
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tmp7 / tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp21 + tmp43
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tmp51 = tmp50 + tmp46
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr13 + (tmp53 + 4*tmp49 + 16*x6), None, eviction_policy='evict_last')
    tmp56 = tmp54 - tmp55
    tmp58 = tmp57 + tmp4
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tmp7 / tmp59
    tmp61 = tmp60 * tmp9
    tmp62 = tmp56 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp44 + tmp66
    tmp69 = tl.full([XBLOCK], 2, tl.int32)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp68 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp68)
    tmp74 = tmp73 + tmp69
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tmp77 = tl.load(in_ptr19 + (tmp76 + 2*tmp72 + 4*x6), None, eviction_policy='evict_last')
    tmp79 = tmp77 - tmp78
    tmp81 = tmp80 + tmp4
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tmp7 / tmp82
    tmp84 = tmp83 * tmp9
    tmp85 = tmp79 * tmp84
    tmp87 = tmp85 * tmp86
    tmp89 = tmp87 + tmp88
    tmp90 = tmp67 + tmp89
    tmp91 = triton_helpers.maximum(tmp18, tmp90)
    tmp92 = tmp19 <= tmp20
    tl.store(in_out_ptr0 + (x5), tmp91, None)
    tl.store(out_ptr1 + (x5), tmp92, None)
''', device_str='cuda')


# kernel path: inductor_cache/og/cog3qgl5xc6eptx4oa2titxmvwyrgb5t6kdzwxecovqy7yn6wg46.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_6 => convolution_292
# Graph fragment:
#   %convolution_292 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_260, %primals_1462, %primals_1463, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 17)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463 = args
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
    assert_size_stride(primals_77, (32, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_78, (32, ), (1, ))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, ), (1, ))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, ), (1, ))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, ), (1, ))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_113, (32, ), (1, ))
    assert_size_stride(primals_114, (32, ), (1, ))
    assert_size_stride(primals_115, (32, ), (1, ))
    assert_size_stride(primals_116, (32, ), (1, ))
    assert_size_stride(primals_117, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (32, ), (1, ))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (32, ), (1, ))
    assert_size_stride(primals_127, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, ), (1, ))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, ), (1, ))
    assert_size_stride(primals_147, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_158, (64, ), (1, ))
    assert_size_stride(primals_159, (64, ), (1, ))
    assert_size_stride(primals_160, (64, ), (1, ))
    assert_size_stride(primals_161, (64, ), (1, ))
    assert_size_stride(primals_162, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (64, ), (1, ))
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_168, (32, ), (1, ))
    assert_size_stride(primals_169, (32, ), (1, ))
    assert_size_stride(primals_170, (32, ), (1, ))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, ), (1, ))
    assert_size_stride(primals_177, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_183, (32, ), (1, ))
    assert_size_stride(primals_184, (32, ), (1, ))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_188, (32, ), (1, ))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (32, ), (1, ))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_218, (32, ), (1, ))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (64, ), (1, ))
    assert_size_stride(primals_226, (64, ), (1, ))
    assert_size_stride(primals_227, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (64, ), (1, ))
    assert_size_stride(primals_232, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (64, ), (1, ))
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, ), (1, ))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_248, (64, ), (1, ))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, ), (1, ))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_273, (128, ), (1, ))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (128, ), (1, ))
    assert_size_stride(primals_277, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, ), (1, ))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_288, (128, ), (1, ))
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (128, ), (1, ))
    assert_size_stride(primals_292, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (128, ), (1, ))
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (32, ), (1, ))
    assert_size_stride(primals_307, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_308, (32, ), (1, ))
    assert_size_stride(primals_309, (32, ), (1, ))
    assert_size_stride(primals_310, (32, ), (1, ))
    assert_size_stride(primals_311, (32, ), (1, ))
    assert_size_stride(primals_312, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, ), (1, ))
    assert_size_stride(primals_321, (64, ), (1, ))
    assert_size_stride(primals_322, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_323, (32, ), (1, ))
    assert_size_stride(primals_324, (32, ), (1, ))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, ), (1, ))
    assert_size_stride(primals_327, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (128, ), (1, ))
    assert_size_stride(primals_337, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_338, (32, ), (1, ))
    assert_size_stride(primals_339, (32, ), (1, ))
    assert_size_stride(primals_340, (32, ), (1, ))
    assert_size_stride(primals_341, (32, ), (1, ))
    assert_size_stride(primals_342, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_343, (32, ), (1, ))
    assert_size_stride(primals_344, (32, ), (1, ))
    assert_size_stride(primals_345, (32, ), (1, ))
    assert_size_stride(primals_346, (32, ), (1, ))
    assert_size_stride(primals_347, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_348, (32, ), (1, ))
    assert_size_stride(primals_349, (32, ), (1, ))
    assert_size_stride(primals_350, (32, ), (1, ))
    assert_size_stride(primals_351, (32, ), (1, ))
    assert_size_stride(primals_352, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_353, (32, ), (1, ))
    assert_size_stride(primals_354, (32, ), (1, ))
    assert_size_stride(primals_355, (32, ), (1, ))
    assert_size_stride(primals_356, (32, ), (1, ))
    assert_size_stride(primals_357, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_358, (32, ), (1, ))
    assert_size_stride(primals_359, (32, ), (1, ))
    assert_size_stride(primals_360, (32, ), (1, ))
    assert_size_stride(primals_361, (32, ), (1, ))
    assert_size_stride(primals_362, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_363, (32, ), (1, ))
    assert_size_stride(primals_364, (32, ), (1, ))
    assert_size_stride(primals_365, (32, ), (1, ))
    assert_size_stride(primals_366, (32, ), (1, ))
    assert_size_stride(primals_367, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_368, (32, ), (1, ))
    assert_size_stride(primals_369, (32, ), (1, ))
    assert_size_stride(primals_370, (32, ), (1, ))
    assert_size_stride(primals_371, (32, ), (1, ))
    assert_size_stride(primals_372, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_373, (32, ), (1, ))
    assert_size_stride(primals_374, (32, ), (1, ))
    assert_size_stride(primals_375, (32, ), (1, ))
    assert_size_stride(primals_376, (32, ), (1, ))
    assert_size_stride(primals_377, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_378, (64, ), (1, ))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (64, ), (1, ))
    assert_size_stride(primals_381, (64, ), (1, ))
    assert_size_stride(primals_382, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (64, ), (1, ))
    assert_size_stride(primals_385, (64, ), (1, ))
    assert_size_stride(primals_386, (64, ), (1, ))
    assert_size_stride(primals_387, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_388, (64, ), (1, ))
    assert_size_stride(primals_389, (64, ), (1, ))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, ), (1, ))
    assert_size_stride(primals_395, (64, ), (1, ))
    assert_size_stride(primals_396, (64, ), (1, ))
    assert_size_stride(primals_397, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_398, (64, ), (1, ))
    assert_size_stride(primals_399, (64, ), (1, ))
    assert_size_stride(primals_400, (64, ), (1, ))
    assert_size_stride(primals_401, (64, ), (1, ))
    assert_size_stride(primals_402, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_403, (64, ), (1, ))
    assert_size_stride(primals_404, (64, ), (1, ))
    assert_size_stride(primals_405, (64, ), (1, ))
    assert_size_stride(primals_406, (64, ), (1, ))
    assert_size_stride(primals_407, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_408, (64, ), (1, ))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, ), (1, ))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_413, (64, ), (1, ))
    assert_size_stride(primals_414, (64, ), (1, ))
    assert_size_stride(primals_415, (64, ), (1, ))
    assert_size_stride(primals_416, (64, ), (1, ))
    assert_size_stride(primals_417, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (128, ), (1, ))
    assert_size_stride(primals_420, (128, ), (1, ))
    assert_size_stride(primals_421, (128, ), (1, ))
    assert_size_stride(primals_422, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, ), (1, ))
    assert_size_stride(primals_425, (128, ), (1, ))
    assert_size_stride(primals_426, (128, ), (1, ))
    assert_size_stride(primals_427, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_428, (128, ), (1, ))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (128, ), (1, ))
    assert_size_stride(primals_437, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (128, ), (1, ))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_443, (128, ), (1, ))
    assert_size_stride(primals_444, (128, ), (1, ))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, ), (1, ))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, ), (1, ))
    assert_size_stride(primals_455, (128, ), (1, ))
    assert_size_stride(primals_456, (128, ), (1, ))
    assert_size_stride(primals_457, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_458, (32, ), (1, ))
    assert_size_stride(primals_459, (32, ), (1, ))
    assert_size_stride(primals_460, (32, ), (1, ))
    assert_size_stride(primals_461, (32, ), (1, ))
    assert_size_stride(primals_462, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_463, (32, ), (1, ))
    assert_size_stride(primals_464, (32, ), (1, ))
    assert_size_stride(primals_465, (32, ), (1, ))
    assert_size_stride(primals_466, (32, ), (1, ))
    assert_size_stride(primals_467, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_468, (64, ), (1, ))
    assert_size_stride(primals_469, (64, ), (1, ))
    assert_size_stride(primals_470, (64, ), (1, ))
    assert_size_stride(primals_471, (64, ), (1, ))
    assert_size_stride(primals_472, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_473, (64, ), (1, ))
    assert_size_stride(primals_474, (64, ), (1, ))
    assert_size_stride(primals_475, (64, ), (1, ))
    assert_size_stride(primals_476, (64, ), (1, ))
    assert_size_stride(primals_477, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_478, (32, ), (1, ))
    assert_size_stride(primals_479, (32, ), (1, ))
    assert_size_stride(primals_480, (32, ), (1, ))
    assert_size_stride(primals_481, (32, ), (1, ))
    assert_size_stride(primals_482, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_483, (128, ), (1, ))
    assert_size_stride(primals_484, (128, ), (1, ))
    assert_size_stride(primals_485, (128, ), (1, ))
    assert_size_stride(primals_486, (128, ), (1, ))
    assert_size_stride(primals_487, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_488, (128, ), (1, ))
    assert_size_stride(primals_489, (128, ), (1, ))
    assert_size_stride(primals_490, (128, ), (1, ))
    assert_size_stride(primals_491, (128, ), (1, ))
    assert_size_stride(primals_492, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_493, (32, ), (1, ))
    assert_size_stride(primals_494, (32, ), (1, ))
    assert_size_stride(primals_495, (32, ), (1, ))
    assert_size_stride(primals_496, (32, ), (1, ))
    assert_size_stride(primals_497, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_498, (32, ), (1, ))
    assert_size_stride(primals_499, (32, ), (1, ))
    assert_size_stride(primals_500, (32, ), (1, ))
    assert_size_stride(primals_501, (32, ), (1, ))
    assert_size_stride(primals_502, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_503, (32, ), (1, ))
    assert_size_stride(primals_504, (32, ), (1, ))
    assert_size_stride(primals_505, (32, ), (1, ))
    assert_size_stride(primals_506, (32, ), (1, ))
    assert_size_stride(primals_507, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_508, (32, ), (1, ))
    assert_size_stride(primals_509, (32, ), (1, ))
    assert_size_stride(primals_510, (32, ), (1, ))
    assert_size_stride(primals_511, (32, ), (1, ))
    assert_size_stride(primals_512, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_513, (32, ), (1, ))
    assert_size_stride(primals_514, (32, ), (1, ))
    assert_size_stride(primals_515, (32, ), (1, ))
    assert_size_stride(primals_516, (32, ), (1, ))
    assert_size_stride(primals_517, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_518, (32, ), (1, ))
    assert_size_stride(primals_519, (32, ), (1, ))
    assert_size_stride(primals_520, (32, ), (1, ))
    assert_size_stride(primals_521, (32, ), (1, ))
    assert_size_stride(primals_522, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_523, (32, ), (1, ))
    assert_size_stride(primals_524, (32, ), (1, ))
    assert_size_stride(primals_525, (32, ), (1, ))
    assert_size_stride(primals_526, (32, ), (1, ))
    assert_size_stride(primals_527, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_528, (32, ), (1, ))
    assert_size_stride(primals_529, (32, ), (1, ))
    assert_size_stride(primals_530, (32, ), (1, ))
    assert_size_stride(primals_531, (32, ), (1, ))
    assert_size_stride(primals_532, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_533, (64, ), (1, ))
    assert_size_stride(primals_534, (64, ), (1, ))
    assert_size_stride(primals_535, (64, ), (1, ))
    assert_size_stride(primals_536, (64, ), (1, ))
    assert_size_stride(primals_537, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_538, (64, ), (1, ))
    assert_size_stride(primals_539, (64, ), (1, ))
    assert_size_stride(primals_540, (64, ), (1, ))
    assert_size_stride(primals_541, (64, ), (1, ))
    assert_size_stride(primals_542, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_543, (64, ), (1, ))
    assert_size_stride(primals_544, (64, ), (1, ))
    assert_size_stride(primals_545, (64, ), (1, ))
    assert_size_stride(primals_546, (64, ), (1, ))
    assert_size_stride(primals_547, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_548, (64, ), (1, ))
    assert_size_stride(primals_549, (64, ), (1, ))
    assert_size_stride(primals_550, (64, ), (1, ))
    assert_size_stride(primals_551, (64, ), (1, ))
    assert_size_stride(primals_552, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_553, (64, ), (1, ))
    assert_size_stride(primals_554, (64, ), (1, ))
    assert_size_stride(primals_555, (64, ), (1, ))
    assert_size_stride(primals_556, (64, ), (1, ))
    assert_size_stride(primals_557, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_558, (64, ), (1, ))
    assert_size_stride(primals_559, (64, ), (1, ))
    assert_size_stride(primals_560, (64, ), (1, ))
    assert_size_stride(primals_561, (64, ), (1, ))
    assert_size_stride(primals_562, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_563, (64, ), (1, ))
    assert_size_stride(primals_564, (64, ), (1, ))
    assert_size_stride(primals_565, (64, ), (1, ))
    assert_size_stride(primals_566, (64, ), (1, ))
    assert_size_stride(primals_567, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_568, (64, ), (1, ))
    assert_size_stride(primals_569, (64, ), (1, ))
    assert_size_stride(primals_570, (64, ), (1, ))
    assert_size_stride(primals_571, (64, ), (1, ))
    assert_size_stride(primals_572, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_573, (128, ), (1, ))
    assert_size_stride(primals_574, (128, ), (1, ))
    assert_size_stride(primals_575, (128, ), (1, ))
    assert_size_stride(primals_576, (128, ), (1, ))
    assert_size_stride(primals_577, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_578, (128, ), (1, ))
    assert_size_stride(primals_579, (128, ), (1, ))
    assert_size_stride(primals_580, (128, ), (1, ))
    assert_size_stride(primals_581, (128, ), (1, ))
    assert_size_stride(primals_582, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_583, (128, ), (1, ))
    assert_size_stride(primals_584, (128, ), (1, ))
    assert_size_stride(primals_585, (128, ), (1, ))
    assert_size_stride(primals_586, (128, ), (1, ))
    assert_size_stride(primals_587, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_588, (128, ), (1, ))
    assert_size_stride(primals_589, (128, ), (1, ))
    assert_size_stride(primals_590, (128, ), (1, ))
    assert_size_stride(primals_591, (128, ), (1, ))
    assert_size_stride(primals_592, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_593, (128, ), (1, ))
    assert_size_stride(primals_594, (128, ), (1, ))
    assert_size_stride(primals_595, (128, ), (1, ))
    assert_size_stride(primals_596, (128, ), (1, ))
    assert_size_stride(primals_597, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_598, (128, ), (1, ))
    assert_size_stride(primals_599, (128, ), (1, ))
    assert_size_stride(primals_600, (128, ), (1, ))
    assert_size_stride(primals_601, (128, ), (1, ))
    assert_size_stride(primals_602, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_603, (128, ), (1, ))
    assert_size_stride(primals_604, (128, ), (1, ))
    assert_size_stride(primals_605, (128, ), (1, ))
    assert_size_stride(primals_606, (128, ), (1, ))
    assert_size_stride(primals_607, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_608, (128, ), (1, ))
    assert_size_stride(primals_609, (128, ), (1, ))
    assert_size_stride(primals_610, (128, ), (1, ))
    assert_size_stride(primals_611, (128, ), (1, ))
    assert_size_stride(primals_612, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_613, (32, ), (1, ))
    assert_size_stride(primals_614, (32, ), (1, ))
    assert_size_stride(primals_615, (32, ), (1, ))
    assert_size_stride(primals_616, (32, ), (1, ))
    assert_size_stride(primals_617, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_618, (32, ), (1, ))
    assert_size_stride(primals_619, (32, ), (1, ))
    assert_size_stride(primals_620, (32, ), (1, ))
    assert_size_stride(primals_621, (32, ), (1, ))
    assert_size_stride(primals_622, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_623, (64, ), (1, ))
    assert_size_stride(primals_624, (64, ), (1, ))
    assert_size_stride(primals_625, (64, ), (1, ))
    assert_size_stride(primals_626, (64, ), (1, ))
    assert_size_stride(primals_627, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_628, (64, ), (1, ))
    assert_size_stride(primals_629, (64, ), (1, ))
    assert_size_stride(primals_630, (64, ), (1, ))
    assert_size_stride(primals_631, (64, ), (1, ))
    assert_size_stride(primals_632, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_633, (32, ), (1, ))
    assert_size_stride(primals_634, (32, ), (1, ))
    assert_size_stride(primals_635, (32, ), (1, ))
    assert_size_stride(primals_636, (32, ), (1, ))
    assert_size_stride(primals_637, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_638, (128, ), (1, ))
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (128, ), (1, ))
    assert_size_stride(primals_641, (128, ), (1, ))
    assert_size_stride(primals_642, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_643, (128, ), (1, ))
    assert_size_stride(primals_644, (128, ), (1, ))
    assert_size_stride(primals_645, (128, ), (1, ))
    assert_size_stride(primals_646, (128, ), (1, ))
    assert_size_stride(primals_647, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_648, (32, ), (1, ))
    assert_size_stride(primals_649, (32, ), (1, ))
    assert_size_stride(primals_650, (32, ), (1, ))
    assert_size_stride(primals_651, (32, ), (1, ))
    assert_size_stride(primals_652, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_653, (32, ), (1, ))
    assert_size_stride(primals_654, (32, ), (1, ))
    assert_size_stride(primals_655, (32, ), (1, ))
    assert_size_stride(primals_656, (32, ), (1, ))
    assert_size_stride(primals_657, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_658, (32, ), (1, ))
    assert_size_stride(primals_659, (32, ), (1, ))
    assert_size_stride(primals_660, (32, ), (1, ))
    assert_size_stride(primals_661, (32, ), (1, ))
    assert_size_stride(primals_662, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_663, (32, ), (1, ))
    assert_size_stride(primals_664, (32, ), (1, ))
    assert_size_stride(primals_665, (32, ), (1, ))
    assert_size_stride(primals_666, (32, ), (1, ))
    assert_size_stride(primals_667, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_668, (32, ), (1, ))
    assert_size_stride(primals_669, (32, ), (1, ))
    assert_size_stride(primals_670, (32, ), (1, ))
    assert_size_stride(primals_671, (32, ), (1, ))
    assert_size_stride(primals_672, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_673, (32, ), (1, ))
    assert_size_stride(primals_674, (32, ), (1, ))
    assert_size_stride(primals_675, (32, ), (1, ))
    assert_size_stride(primals_676, (32, ), (1, ))
    assert_size_stride(primals_677, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_678, (32, ), (1, ))
    assert_size_stride(primals_679, (32, ), (1, ))
    assert_size_stride(primals_680, (32, ), (1, ))
    assert_size_stride(primals_681, (32, ), (1, ))
    assert_size_stride(primals_682, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_683, (32, ), (1, ))
    assert_size_stride(primals_684, (32, ), (1, ))
    assert_size_stride(primals_685, (32, ), (1, ))
    assert_size_stride(primals_686, (32, ), (1, ))
    assert_size_stride(primals_687, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_688, (64, ), (1, ))
    assert_size_stride(primals_689, (64, ), (1, ))
    assert_size_stride(primals_690, (64, ), (1, ))
    assert_size_stride(primals_691, (64, ), (1, ))
    assert_size_stride(primals_692, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_693, (64, ), (1, ))
    assert_size_stride(primals_694, (64, ), (1, ))
    assert_size_stride(primals_695, (64, ), (1, ))
    assert_size_stride(primals_696, (64, ), (1, ))
    assert_size_stride(primals_697, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_698, (64, ), (1, ))
    assert_size_stride(primals_699, (64, ), (1, ))
    assert_size_stride(primals_700, (64, ), (1, ))
    assert_size_stride(primals_701, (64, ), (1, ))
    assert_size_stride(primals_702, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_703, (64, ), (1, ))
    assert_size_stride(primals_704, (64, ), (1, ))
    assert_size_stride(primals_705, (64, ), (1, ))
    assert_size_stride(primals_706, (64, ), (1, ))
    assert_size_stride(primals_707, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_708, (64, ), (1, ))
    assert_size_stride(primals_709, (64, ), (1, ))
    assert_size_stride(primals_710, (64, ), (1, ))
    assert_size_stride(primals_711, (64, ), (1, ))
    assert_size_stride(primals_712, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_713, (64, ), (1, ))
    assert_size_stride(primals_714, (64, ), (1, ))
    assert_size_stride(primals_715, (64, ), (1, ))
    assert_size_stride(primals_716, (64, ), (1, ))
    assert_size_stride(primals_717, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_718, (64, ), (1, ))
    assert_size_stride(primals_719, (64, ), (1, ))
    assert_size_stride(primals_720, (64, ), (1, ))
    assert_size_stride(primals_721, (64, ), (1, ))
    assert_size_stride(primals_722, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_723, (64, ), (1, ))
    assert_size_stride(primals_724, (64, ), (1, ))
    assert_size_stride(primals_725, (64, ), (1, ))
    assert_size_stride(primals_726, (64, ), (1, ))
    assert_size_stride(primals_727, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_728, (128, ), (1, ))
    assert_size_stride(primals_729, (128, ), (1, ))
    assert_size_stride(primals_730, (128, ), (1, ))
    assert_size_stride(primals_731, (128, ), (1, ))
    assert_size_stride(primals_732, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_733, (128, ), (1, ))
    assert_size_stride(primals_734, (128, ), (1, ))
    assert_size_stride(primals_735, (128, ), (1, ))
    assert_size_stride(primals_736, (128, ), (1, ))
    assert_size_stride(primals_737, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_738, (128, ), (1, ))
    assert_size_stride(primals_739, (128, ), (1, ))
    assert_size_stride(primals_740, (128, ), (1, ))
    assert_size_stride(primals_741, (128, ), (1, ))
    assert_size_stride(primals_742, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_743, (128, ), (1, ))
    assert_size_stride(primals_744, (128, ), (1, ))
    assert_size_stride(primals_745, (128, ), (1, ))
    assert_size_stride(primals_746, (128, ), (1, ))
    assert_size_stride(primals_747, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_748, (128, ), (1, ))
    assert_size_stride(primals_749, (128, ), (1, ))
    assert_size_stride(primals_750, (128, ), (1, ))
    assert_size_stride(primals_751, (128, ), (1, ))
    assert_size_stride(primals_752, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_753, (128, ), (1, ))
    assert_size_stride(primals_754, (128, ), (1, ))
    assert_size_stride(primals_755, (128, ), (1, ))
    assert_size_stride(primals_756, (128, ), (1, ))
    assert_size_stride(primals_757, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_758, (128, ), (1, ))
    assert_size_stride(primals_759, (128, ), (1, ))
    assert_size_stride(primals_760, (128, ), (1, ))
    assert_size_stride(primals_761, (128, ), (1, ))
    assert_size_stride(primals_762, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_763, (128, ), (1, ))
    assert_size_stride(primals_764, (128, ), (1, ))
    assert_size_stride(primals_765, (128, ), (1, ))
    assert_size_stride(primals_766, (128, ), (1, ))
    assert_size_stride(primals_767, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_768, (32, ), (1, ))
    assert_size_stride(primals_769, (32, ), (1, ))
    assert_size_stride(primals_770, (32, ), (1, ))
    assert_size_stride(primals_771, (32, ), (1, ))
    assert_size_stride(primals_772, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_773, (32, ), (1, ))
    assert_size_stride(primals_774, (32, ), (1, ))
    assert_size_stride(primals_775, (32, ), (1, ))
    assert_size_stride(primals_776, (32, ), (1, ))
    assert_size_stride(primals_777, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_778, (64, ), (1, ))
    assert_size_stride(primals_779, (64, ), (1, ))
    assert_size_stride(primals_780, (64, ), (1, ))
    assert_size_stride(primals_781, (64, ), (1, ))
    assert_size_stride(primals_782, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_783, (64, ), (1, ))
    assert_size_stride(primals_784, (64, ), (1, ))
    assert_size_stride(primals_785, (64, ), (1, ))
    assert_size_stride(primals_786, (64, ), (1, ))
    assert_size_stride(primals_787, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_788, (32, ), (1, ))
    assert_size_stride(primals_789, (32, ), (1, ))
    assert_size_stride(primals_790, (32, ), (1, ))
    assert_size_stride(primals_791, (32, ), (1, ))
    assert_size_stride(primals_792, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_793, (128, ), (1, ))
    assert_size_stride(primals_794, (128, ), (1, ))
    assert_size_stride(primals_795, (128, ), (1, ))
    assert_size_stride(primals_796, (128, ), (1, ))
    assert_size_stride(primals_797, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_798, (128, ), (1, ))
    assert_size_stride(primals_799, (128, ), (1, ))
    assert_size_stride(primals_800, (128, ), (1, ))
    assert_size_stride(primals_801, (128, ), (1, ))
    assert_size_stride(primals_802, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_803, (256, ), (1, ))
    assert_size_stride(primals_804, (256, ), (1, ))
    assert_size_stride(primals_805, (256, ), (1, ))
    assert_size_stride(primals_806, (256, ), (1, ))
    assert_size_stride(primals_807, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_808, (32, ), (1, ))
    assert_size_stride(primals_809, (32, ), (1, ))
    assert_size_stride(primals_810, (32, ), (1, ))
    assert_size_stride(primals_811, (32, ), (1, ))
    assert_size_stride(primals_812, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_813, (32, ), (1, ))
    assert_size_stride(primals_814, (32, ), (1, ))
    assert_size_stride(primals_815, (32, ), (1, ))
    assert_size_stride(primals_816, (32, ), (1, ))
    assert_size_stride(primals_817, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_818, (32, ), (1, ))
    assert_size_stride(primals_819, (32, ), (1, ))
    assert_size_stride(primals_820, (32, ), (1, ))
    assert_size_stride(primals_821, (32, ), (1, ))
    assert_size_stride(primals_822, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_823, (32, ), (1, ))
    assert_size_stride(primals_824, (32, ), (1, ))
    assert_size_stride(primals_825, (32, ), (1, ))
    assert_size_stride(primals_826, (32, ), (1, ))
    assert_size_stride(primals_827, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_828, (32, ), (1, ))
    assert_size_stride(primals_829, (32, ), (1, ))
    assert_size_stride(primals_830, (32, ), (1, ))
    assert_size_stride(primals_831, (32, ), (1, ))
    assert_size_stride(primals_832, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_833, (32, ), (1, ))
    assert_size_stride(primals_834, (32, ), (1, ))
    assert_size_stride(primals_835, (32, ), (1, ))
    assert_size_stride(primals_836, (32, ), (1, ))
    assert_size_stride(primals_837, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_838, (32, ), (1, ))
    assert_size_stride(primals_839, (32, ), (1, ))
    assert_size_stride(primals_840, (32, ), (1, ))
    assert_size_stride(primals_841, (32, ), (1, ))
    assert_size_stride(primals_842, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_843, (32, ), (1, ))
    assert_size_stride(primals_844, (32, ), (1, ))
    assert_size_stride(primals_845, (32, ), (1, ))
    assert_size_stride(primals_846, (32, ), (1, ))
    assert_size_stride(primals_847, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_848, (64, ), (1, ))
    assert_size_stride(primals_849, (64, ), (1, ))
    assert_size_stride(primals_850, (64, ), (1, ))
    assert_size_stride(primals_851, (64, ), (1, ))
    assert_size_stride(primals_852, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_853, (64, ), (1, ))
    assert_size_stride(primals_854, (64, ), (1, ))
    assert_size_stride(primals_855, (64, ), (1, ))
    assert_size_stride(primals_856, (64, ), (1, ))
    assert_size_stride(primals_857, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_858, (64, ), (1, ))
    assert_size_stride(primals_859, (64, ), (1, ))
    assert_size_stride(primals_860, (64, ), (1, ))
    assert_size_stride(primals_861, (64, ), (1, ))
    assert_size_stride(primals_862, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_863, (64, ), (1, ))
    assert_size_stride(primals_864, (64, ), (1, ))
    assert_size_stride(primals_865, (64, ), (1, ))
    assert_size_stride(primals_866, (64, ), (1, ))
    assert_size_stride(primals_867, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_868, (64, ), (1, ))
    assert_size_stride(primals_869, (64, ), (1, ))
    assert_size_stride(primals_870, (64, ), (1, ))
    assert_size_stride(primals_871, (64, ), (1, ))
    assert_size_stride(primals_872, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_873, (64, ), (1, ))
    assert_size_stride(primals_874, (64, ), (1, ))
    assert_size_stride(primals_875, (64, ), (1, ))
    assert_size_stride(primals_876, (64, ), (1, ))
    assert_size_stride(primals_877, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_878, (64, ), (1, ))
    assert_size_stride(primals_879, (64, ), (1, ))
    assert_size_stride(primals_880, (64, ), (1, ))
    assert_size_stride(primals_881, (64, ), (1, ))
    assert_size_stride(primals_882, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_883, (64, ), (1, ))
    assert_size_stride(primals_884, (64, ), (1, ))
    assert_size_stride(primals_885, (64, ), (1, ))
    assert_size_stride(primals_886, (64, ), (1, ))
    assert_size_stride(primals_887, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_888, (128, ), (1, ))
    assert_size_stride(primals_889, (128, ), (1, ))
    assert_size_stride(primals_890, (128, ), (1, ))
    assert_size_stride(primals_891, (128, ), (1, ))
    assert_size_stride(primals_892, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_893, (128, ), (1, ))
    assert_size_stride(primals_894, (128, ), (1, ))
    assert_size_stride(primals_895, (128, ), (1, ))
    assert_size_stride(primals_896, (128, ), (1, ))
    assert_size_stride(primals_897, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_898, (128, ), (1, ))
    assert_size_stride(primals_899, (128, ), (1, ))
    assert_size_stride(primals_900, (128, ), (1, ))
    assert_size_stride(primals_901, (128, ), (1, ))
    assert_size_stride(primals_902, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_903, (128, ), (1, ))
    assert_size_stride(primals_904, (128, ), (1, ))
    assert_size_stride(primals_905, (128, ), (1, ))
    assert_size_stride(primals_906, (128, ), (1, ))
    assert_size_stride(primals_907, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_908, (128, ), (1, ))
    assert_size_stride(primals_909, (128, ), (1, ))
    assert_size_stride(primals_910, (128, ), (1, ))
    assert_size_stride(primals_911, (128, ), (1, ))
    assert_size_stride(primals_912, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_913, (128, ), (1, ))
    assert_size_stride(primals_914, (128, ), (1, ))
    assert_size_stride(primals_915, (128, ), (1, ))
    assert_size_stride(primals_916, (128, ), (1, ))
    assert_size_stride(primals_917, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_918, (128, ), (1, ))
    assert_size_stride(primals_919, (128, ), (1, ))
    assert_size_stride(primals_920, (128, ), (1, ))
    assert_size_stride(primals_921, (128, ), (1, ))
    assert_size_stride(primals_922, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_923, (128, ), (1, ))
    assert_size_stride(primals_924, (128, ), (1, ))
    assert_size_stride(primals_925, (128, ), (1, ))
    assert_size_stride(primals_926, (128, ), (1, ))
    assert_size_stride(primals_927, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_928, (256, ), (1, ))
    assert_size_stride(primals_929, (256, ), (1, ))
    assert_size_stride(primals_930, (256, ), (1, ))
    assert_size_stride(primals_931, (256, ), (1, ))
    assert_size_stride(primals_932, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_933, (256, ), (1, ))
    assert_size_stride(primals_934, (256, ), (1, ))
    assert_size_stride(primals_935, (256, ), (1, ))
    assert_size_stride(primals_936, (256, ), (1, ))
    assert_size_stride(primals_937, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_938, (256, ), (1, ))
    assert_size_stride(primals_939, (256, ), (1, ))
    assert_size_stride(primals_940, (256, ), (1, ))
    assert_size_stride(primals_941, (256, ), (1, ))
    assert_size_stride(primals_942, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_943, (256, ), (1, ))
    assert_size_stride(primals_944, (256, ), (1, ))
    assert_size_stride(primals_945, (256, ), (1, ))
    assert_size_stride(primals_946, (256, ), (1, ))
    assert_size_stride(primals_947, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_948, (256, ), (1, ))
    assert_size_stride(primals_949, (256, ), (1, ))
    assert_size_stride(primals_950, (256, ), (1, ))
    assert_size_stride(primals_951, (256, ), (1, ))
    assert_size_stride(primals_952, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_953, (256, ), (1, ))
    assert_size_stride(primals_954, (256, ), (1, ))
    assert_size_stride(primals_955, (256, ), (1, ))
    assert_size_stride(primals_956, (256, ), (1, ))
    assert_size_stride(primals_957, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_958, (256, ), (1, ))
    assert_size_stride(primals_959, (256, ), (1, ))
    assert_size_stride(primals_960, (256, ), (1, ))
    assert_size_stride(primals_961, (256, ), (1, ))
    assert_size_stride(primals_962, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_963, (256, ), (1, ))
    assert_size_stride(primals_964, (256, ), (1, ))
    assert_size_stride(primals_965, (256, ), (1, ))
    assert_size_stride(primals_966, (256, ), (1, ))
    assert_size_stride(primals_967, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_968, (32, ), (1, ))
    assert_size_stride(primals_969, (32, ), (1, ))
    assert_size_stride(primals_970, (32, ), (1, ))
    assert_size_stride(primals_971, (32, ), (1, ))
    assert_size_stride(primals_972, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_973, (32, ), (1, ))
    assert_size_stride(primals_974, (32, ), (1, ))
    assert_size_stride(primals_975, (32, ), (1, ))
    assert_size_stride(primals_976, (32, ), (1, ))
    assert_size_stride(primals_977, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_978, (32, ), (1, ))
    assert_size_stride(primals_979, (32, ), (1, ))
    assert_size_stride(primals_980, (32, ), (1, ))
    assert_size_stride(primals_981, (32, ), (1, ))
    assert_size_stride(primals_982, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_983, (64, ), (1, ))
    assert_size_stride(primals_984, (64, ), (1, ))
    assert_size_stride(primals_985, (64, ), (1, ))
    assert_size_stride(primals_986, (64, ), (1, ))
    assert_size_stride(primals_987, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_988, (64, ), (1, ))
    assert_size_stride(primals_989, (64, ), (1, ))
    assert_size_stride(primals_990, (64, ), (1, ))
    assert_size_stride(primals_991, (64, ), (1, ))
    assert_size_stride(primals_992, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_993, (64, ), (1, ))
    assert_size_stride(primals_994, (64, ), (1, ))
    assert_size_stride(primals_995, (64, ), (1, ))
    assert_size_stride(primals_996, (64, ), (1, ))
    assert_size_stride(primals_997, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_998, (32, ), (1, ))
    assert_size_stride(primals_999, (32, ), (1, ))
    assert_size_stride(primals_1000, (32, ), (1, ))
    assert_size_stride(primals_1001, (32, ), (1, ))
    assert_size_stride(primals_1002, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1003, (128, ), (1, ))
    assert_size_stride(primals_1004, (128, ), (1, ))
    assert_size_stride(primals_1005, (128, ), (1, ))
    assert_size_stride(primals_1006, (128, ), (1, ))
    assert_size_stride(primals_1007, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1008, (128, ), (1, ))
    assert_size_stride(primals_1009, (128, ), (1, ))
    assert_size_stride(primals_1010, (128, ), (1, ))
    assert_size_stride(primals_1011, (128, ), (1, ))
    assert_size_stride(primals_1012, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1013, (128, ), (1, ))
    assert_size_stride(primals_1014, (128, ), (1, ))
    assert_size_stride(primals_1015, (128, ), (1, ))
    assert_size_stride(primals_1016, (128, ), (1, ))
    assert_size_stride(primals_1017, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1018, (32, ), (1, ))
    assert_size_stride(primals_1019, (32, ), (1, ))
    assert_size_stride(primals_1020, (32, ), (1, ))
    assert_size_stride(primals_1021, (32, ), (1, ))
    assert_size_stride(primals_1022, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1023, (32, ), (1, ))
    assert_size_stride(primals_1024, (32, ), (1, ))
    assert_size_stride(primals_1025, (32, ), (1, ))
    assert_size_stride(primals_1026, (32, ), (1, ))
    assert_size_stride(primals_1027, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1028, (256, ), (1, ))
    assert_size_stride(primals_1029, (256, ), (1, ))
    assert_size_stride(primals_1030, (256, ), (1, ))
    assert_size_stride(primals_1031, (256, ), (1, ))
    assert_size_stride(primals_1032, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1033, (64, ), (1, ))
    assert_size_stride(primals_1034, (64, ), (1, ))
    assert_size_stride(primals_1035, (64, ), (1, ))
    assert_size_stride(primals_1036, (64, ), (1, ))
    assert_size_stride(primals_1037, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1038, (256, ), (1, ))
    assert_size_stride(primals_1039, (256, ), (1, ))
    assert_size_stride(primals_1040, (256, ), (1, ))
    assert_size_stride(primals_1041, (256, ), (1, ))
    assert_size_stride(primals_1042, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1043, (256, ), (1, ))
    assert_size_stride(primals_1044, (256, ), (1, ))
    assert_size_stride(primals_1045, (256, ), (1, ))
    assert_size_stride(primals_1046, (256, ), (1, ))
    assert_size_stride(primals_1047, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1048, (32, ), (1, ))
    assert_size_stride(primals_1049, (32, ), (1, ))
    assert_size_stride(primals_1050, (32, ), (1, ))
    assert_size_stride(primals_1051, (32, ), (1, ))
    assert_size_stride(primals_1052, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1053, (32, ), (1, ))
    assert_size_stride(primals_1054, (32, ), (1, ))
    assert_size_stride(primals_1055, (32, ), (1, ))
    assert_size_stride(primals_1056, (32, ), (1, ))
    assert_size_stride(primals_1057, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1058, (32, ), (1, ))
    assert_size_stride(primals_1059, (32, ), (1, ))
    assert_size_stride(primals_1060, (32, ), (1, ))
    assert_size_stride(primals_1061, (32, ), (1, ))
    assert_size_stride(primals_1062, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1063, (32, ), (1, ))
    assert_size_stride(primals_1064, (32, ), (1, ))
    assert_size_stride(primals_1065, (32, ), (1, ))
    assert_size_stride(primals_1066, (32, ), (1, ))
    assert_size_stride(primals_1067, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1068, (32, ), (1, ))
    assert_size_stride(primals_1069, (32, ), (1, ))
    assert_size_stride(primals_1070, (32, ), (1, ))
    assert_size_stride(primals_1071, (32, ), (1, ))
    assert_size_stride(primals_1072, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1073, (32, ), (1, ))
    assert_size_stride(primals_1074, (32, ), (1, ))
    assert_size_stride(primals_1075, (32, ), (1, ))
    assert_size_stride(primals_1076, (32, ), (1, ))
    assert_size_stride(primals_1077, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1078, (32, ), (1, ))
    assert_size_stride(primals_1079, (32, ), (1, ))
    assert_size_stride(primals_1080, (32, ), (1, ))
    assert_size_stride(primals_1081, (32, ), (1, ))
    assert_size_stride(primals_1082, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1083, (32, ), (1, ))
    assert_size_stride(primals_1084, (32, ), (1, ))
    assert_size_stride(primals_1085, (32, ), (1, ))
    assert_size_stride(primals_1086, (32, ), (1, ))
    assert_size_stride(primals_1087, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1088, (64, ), (1, ))
    assert_size_stride(primals_1089, (64, ), (1, ))
    assert_size_stride(primals_1090, (64, ), (1, ))
    assert_size_stride(primals_1091, (64, ), (1, ))
    assert_size_stride(primals_1092, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1093, (64, ), (1, ))
    assert_size_stride(primals_1094, (64, ), (1, ))
    assert_size_stride(primals_1095, (64, ), (1, ))
    assert_size_stride(primals_1096, (64, ), (1, ))
    assert_size_stride(primals_1097, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1098, (64, ), (1, ))
    assert_size_stride(primals_1099, (64, ), (1, ))
    assert_size_stride(primals_1100, (64, ), (1, ))
    assert_size_stride(primals_1101, (64, ), (1, ))
    assert_size_stride(primals_1102, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1103, (64, ), (1, ))
    assert_size_stride(primals_1104, (64, ), (1, ))
    assert_size_stride(primals_1105, (64, ), (1, ))
    assert_size_stride(primals_1106, (64, ), (1, ))
    assert_size_stride(primals_1107, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1108, (64, ), (1, ))
    assert_size_stride(primals_1109, (64, ), (1, ))
    assert_size_stride(primals_1110, (64, ), (1, ))
    assert_size_stride(primals_1111, (64, ), (1, ))
    assert_size_stride(primals_1112, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1113, (64, ), (1, ))
    assert_size_stride(primals_1114, (64, ), (1, ))
    assert_size_stride(primals_1115, (64, ), (1, ))
    assert_size_stride(primals_1116, (64, ), (1, ))
    assert_size_stride(primals_1117, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1118, (64, ), (1, ))
    assert_size_stride(primals_1119, (64, ), (1, ))
    assert_size_stride(primals_1120, (64, ), (1, ))
    assert_size_stride(primals_1121, (64, ), (1, ))
    assert_size_stride(primals_1122, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1123, (64, ), (1, ))
    assert_size_stride(primals_1124, (64, ), (1, ))
    assert_size_stride(primals_1125, (64, ), (1, ))
    assert_size_stride(primals_1126, (64, ), (1, ))
    assert_size_stride(primals_1127, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1128, (128, ), (1, ))
    assert_size_stride(primals_1129, (128, ), (1, ))
    assert_size_stride(primals_1130, (128, ), (1, ))
    assert_size_stride(primals_1131, (128, ), (1, ))
    assert_size_stride(primals_1132, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1133, (128, ), (1, ))
    assert_size_stride(primals_1134, (128, ), (1, ))
    assert_size_stride(primals_1135, (128, ), (1, ))
    assert_size_stride(primals_1136, (128, ), (1, ))
    assert_size_stride(primals_1137, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1138, (128, ), (1, ))
    assert_size_stride(primals_1139, (128, ), (1, ))
    assert_size_stride(primals_1140, (128, ), (1, ))
    assert_size_stride(primals_1141, (128, ), (1, ))
    assert_size_stride(primals_1142, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1143, (128, ), (1, ))
    assert_size_stride(primals_1144, (128, ), (1, ))
    assert_size_stride(primals_1145, (128, ), (1, ))
    assert_size_stride(primals_1146, (128, ), (1, ))
    assert_size_stride(primals_1147, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1148, (128, ), (1, ))
    assert_size_stride(primals_1149, (128, ), (1, ))
    assert_size_stride(primals_1150, (128, ), (1, ))
    assert_size_stride(primals_1151, (128, ), (1, ))
    assert_size_stride(primals_1152, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1153, (128, ), (1, ))
    assert_size_stride(primals_1154, (128, ), (1, ))
    assert_size_stride(primals_1155, (128, ), (1, ))
    assert_size_stride(primals_1156, (128, ), (1, ))
    assert_size_stride(primals_1157, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1158, (128, ), (1, ))
    assert_size_stride(primals_1159, (128, ), (1, ))
    assert_size_stride(primals_1160, (128, ), (1, ))
    assert_size_stride(primals_1161, (128, ), (1, ))
    assert_size_stride(primals_1162, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1163, (128, ), (1, ))
    assert_size_stride(primals_1164, (128, ), (1, ))
    assert_size_stride(primals_1165, (128, ), (1, ))
    assert_size_stride(primals_1166, (128, ), (1, ))
    assert_size_stride(primals_1167, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1168, (256, ), (1, ))
    assert_size_stride(primals_1169, (256, ), (1, ))
    assert_size_stride(primals_1170, (256, ), (1, ))
    assert_size_stride(primals_1171, (256, ), (1, ))
    assert_size_stride(primals_1172, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1173, (256, ), (1, ))
    assert_size_stride(primals_1174, (256, ), (1, ))
    assert_size_stride(primals_1175, (256, ), (1, ))
    assert_size_stride(primals_1176, (256, ), (1, ))
    assert_size_stride(primals_1177, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1178, (256, ), (1, ))
    assert_size_stride(primals_1179, (256, ), (1, ))
    assert_size_stride(primals_1180, (256, ), (1, ))
    assert_size_stride(primals_1181, (256, ), (1, ))
    assert_size_stride(primals_1182, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1183, (256, ), (1, ))
    assert_size_stride(primals_1184, (256, ), (1, ))
    assert_size_stride(primals_1185, (256, ), (1, ))
    assert_size_stride(primals_1186, (256, ), (1, ))
    assert_size_stride(primals_1187, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1188, (256, ), (1, ))
    assert_size_stride(primals_1189, (256, ), (1, ))
    assert_size_stride(primals_1190, (256, ), (1, ))
    assert_size_stride(primals_1191, (256, ), (1, ))
    assert_size_stride(primals_1192, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1193, (256, ), (1, ))
    assert_size_stride(primals_1194, (256, ), (1, ))
    assert_size_stride(primals_1195, (256, ), (1, ))
    assert_size_stride(primals_1196, (256, ), (1, ))
    assert_size_stride(primals_1197, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1198, (256, ), (1, ))
    assert_size_stride(primals_1199, (256, ), (1, ))
    assert_size_stride(primals_1200, (256, ), (1, ))
    assert_size_stride(primals_1201, (256, ), (1, ))
    assert_size_stride(primals_1202, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1203, (256, ), (1, ))
    assert_size_stride(primals_1204, (256, ), (1, ))
    assert_size_stride(primals_1205, (256, ), (1, ))
    assert_size_stride(primals_1206, (256, ), (1, ))
    assert_size_stride(primals_1207, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_1208, (32, ), (1, ))
    assert_size_stride(primals_1209, (32, ), (1, ))
    assert_size_stride(primals_1210, (32, ), (1, ))
    assert_size_stride(primals_1211, (32, ), (1, ))
    assert_size_stride(primals_1212, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_1213, (32, ), (1, ))
    assert_size_stride(primals_1214, (32, ), (1, ))
    assert_size_stride(primals_1215, (32, ), (1, ))
    assert_size_stride(primals_1216, (32, ), (1, ))
    assert_size_stride(primals_1217, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1218, (32, ), (1, ))
    assert_size_stride(primals_1219, (32, ), (1, ))
    assert_size_stride(primals_1220, (32, ), (1, ))
    assert_size_stride(primals_1221, (32, ), (1, ))
    assert_size_stride(primals_1222, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1223, (64, ), (1, ))
    assert_size_stride(primals_1224, (64, ), (1, ))
    assert_size_stride(primals_1225, (64, ), (1, ))
    assert_size_stride(primals_1226, (64, ), (1, ))
    assert_size_stride(primals_1227, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_1228, (64, ), (1, ))
    assert_size_stride(primals_1229, (64, ), (1, ))
    assert_size_stride(primals_1230, (64, ), (1, ))
    assert_size_stride(primals_1231, (64, ), (1, ))
    assert_size_stride(primals_1232, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1233, (64, ), (1, ))
    assert_size_stride(primals_1234, (64, ), (1, ))
    assert_size_stride(primals_1235, (64, ), (1, ))
    assert_size_stride(primals_1236, (64, ), (1, ))
    assert_size_stride(primals_1237, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1238, (32, ), (1, ))
    assert_size_stride(primals_1239, (32, ), (1, ))
    assert_size_stride(primals_1240, (32, ), (1, ))
    assert_size_stride(primals_1241, (32, ), (1, ))
    assert_size_stride(primals_1242, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1243, (128, ), (1, ))
    assert_size_stride(primals_1244, (128, ), (1, ))
    assert_size_stride(primals_1245, (128, ), (1, ))
    assert_size_stride(primals_1246, (128, ), (1, ))
    assert_size_stride(primals_1247, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1248, (128, ), (1, ))
    assert_size_stride(primals_1249, (128, ), (1, ))
    assert_size_stride(primals_1250, (128, ), (1, ))
    assert_size_stride(primals_1251, (128, ), (1, ))
    assert_size_stride(primals_1252, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1253, (128, ), (1, ))
    assert_size_stride(primals_1254, (128, ), (1, ))
    assert_size_stride(primals_1255, (128, ), (1, ))
    assert_size_stride(primals_1256, (128, ), (1, ))
    assert_size_stride(primals_1257, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1258, (32, ), (1, ))
    assert_size_stride(primals_1259, (32, ), (1, ))
    assert_size_stride(primals_1260, (32, ), (1, ))
    assert_size_stride(primals_1261, (32, ), (1, ))
    assert_size_stride(primals_1262, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1263, (32, ), (1, ))
    assert_size_stride(primals_1264, (32, ), (1, ))
    assert_size_stride(primals_1265, (32, ), (1, ))
    assert_size_stride(primals_1266, (32, ), (1, ))
    assert_size_stride(primals_1267, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1268, (256, ), (1, ))
    assert_size_stride(primals_1269, (256, ), (1, ))
    assert_size_stride(primals_1270, (256, ), (1, ))
    assert_size_stride(primals_1271, (256, ), (1, ))
    assert_size_stride(primals_1272, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1273, (64, ), (1, ))
    assert_size_stride(primals_1274, (64, ), (1, ))
    assert_size_stride(primals_1275, (64, ), (1, ))
    assert_size_stride(primals_1276, (64, ), (1, ))
    assert_size_stride(primals_1277, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1278, (256, ), (1, ))
    assert_size_stride(primals_1279, (256, ), (1, ))
    assert_size_stride(primals_1280, (256, ), (1, ))
    assert_size_stride(primals_1281, (256, ), (1, ))
    assert_size_stride(primals_1282, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1283, (256, ), (1, ))
    assert_size_stride(primals_1284, (256, ), (1, ))
    assert_size_stride(primals_1285, (256, ), (1, ))
    assert_size_stride(primals_1286, (256, ), (1, ))
    assert_size_stride(primals_1287, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1288, (32, ), (1, ))
    assert_size_stride(primals_1289, (32, ), (1, ))
    assert_size_stride(primals_1290, (32, ), (1, ))
    assert_size_stride(primals_1291, (32, ), (1, ))
    assert_size_stride(primals_1292, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1293, (32, ), (1, ))
    assert_size_stride(primals_1294, (32, ), (1, ))
    assert_size_stride(primals_1295, (32, ), (1, ))
    assert_size_stride(primals_1296, (32, ), (1, ))
    assert_size_stride(primals_1297, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1298, (32, ), (1, ))
    assert_size_stride(primals_1299, (32, ), (1, ))
    assert_size_stride(primals_1300, (32, ), (1, ))
    assert_size_stride(primals_1301, (32, ), (1, ))
    assert_size_stride(primals_1302, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1303, (32, ), (1, ))
    assert_size_stride(primals_1304, (32, ), (1, ))
    assert_size_stride(primals_1305, (32, ), (1, ))
    assert_size_stride(primals_1306, (32, ), (1, ))
    assert_size_stride(primals_1307, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1308, (32, ), (1, ))
    assert_size_stride(primals_1309, (32, ), (1, ))
    assert_size_stride(primals_1310, (32, ), (1, ))
    assert_size_stride(primals_1311, (32, ), (1, ))
    assert_size_stride(primals_1312, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1313, (32, ), (1, ))
    assert_size_stride(primals_1314, (32, ), (1, ))
    assert_size_stride(primals_1315, (32, ), (1, ))
    assert_size_stride(primals_1316, (32, ), (1, ))
    assert_size_stride(primals_1317, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1318, (32, ), (1, ))
    assert_size_stride(primals_1319, (32, ), (1, ))
    assert_size_stride(primals_1320, (32, ), (1, ))
    assert_size_stride(primals_1321, (32, ), (1, ))
    assert_size_stride(primals_1322, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_1323, (32, ), (1, ))
    assert_size_stride(primals_1324, (32, ), (1, ))
    assert_size_stride(primals_1325, (32, ), (1, ))
    assert_size_stride(primals_1326, (32, ), (1, ))
    assert_size_stride(primals_1327, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1328, (64, ), (1, ))
    assert_size_stride(primals_1329, (64, ), (1, ))
    assert_size_stride(primals_1330, (64, ), (1, ))
    assert_size_stride(primals_1331, (64, ), (1, ))
    assert_size_stride(primals_1332, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1333, (64, ), (1, ))
    assert_size_stride(primals_1334, (64, ), (1, ))
    assert_size_stride(primals_1335, (64, ), (1, ))
    assert_size_stride(primals_1336, (64, ), (1, ))
    assert_size_stride(primals_1337, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1338, (64, ), (1, ))
    assert_size_stride(primals_1339, (64, ), (1, ))
    assert_size_stride(primals_1340, (64, ), (1, ))
    assert_size_stride(primals_1341, (64, ), (1, ))
    assert_size_stride(primals_1342, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1343, (64, ), (1, ))
    assert_size_stride(primals_1344, (64, ), (1, ))
    assert_size_stride(primals_1345, (64, ), (1, ))
    assert_size_stride(primals_1346, (64, ), (1, ))
    assert_size_stride(primals_1347, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1348, (64, ), (1, ))
    assert_size_stride(primals_1349, (64, ), (1, ))
    assert_size_stride(primals_1350, (64, ), (1, ))
    assert_size_stride(primals_1351, (64, ), (1, ))
    assert_size_stride(primals_1352, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1353, (64, ), (1, ))
    assert_size_stride(primals_1354, (64, ), (1, ))
    assert_size_stride(primals_1355, (64, ), (1, ))
    assert_size_stride(primals_1356, (64, ), (1, ))
    assert_size_stride(primals_1357, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1358, (64, ), (1, ))
    assert_size_stride(primals_1359, (64, ), (1, ))
    assert_size_stride(primals_1360, (64, ), (1, ))
    assert_size_stride(primals_1361, (64, ), (1, ))
    assert_size_stride(primals_1362, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_1363, (64, ), (1, ))
    assert_size_stride(primals_1364, (64, ), (1, ))
    assert_size_stride(primals_1365, (64, ), (1, ))
    assert_size_stride(primals_1366, (64, ), (1, ))
    assert_size_stride(primals_1367, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1368, (128, ), (1, ))
    assert_size_stride(primals_1369, (128, ), (1, ))
    assert_size_stride(primals_1370, (128, ), (1, ))
    assert_size_stride(primals_1371, (128, ), (1, ))
    assert_size_stride(primals_1372, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1373, (128, ), (1, ))
    assert_size_stride(primals_1374, (128, ), (1, ))
    assert_size_stride(primals_1375, (128, ), (1, ))
    assert_size_stride(primals_1376, (128, ), (1, ))
    assert_size_stride(primals_1377, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1378, (128, ), (1, ))
    assert_size_stride(primals_1379, (128, ), (1, ))
    assert_size_stride(primals_1380, (128, ), (1, ))
    assert_size_stride(primals_1381, (128, ), (1, ))
    assert_size_stride(primals_1382, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1383, (128, ), (1, ))
    assert_size_stride(primals_1384, (128, ), (1, ))
    assert_size_stride(primals_1385, (128, ), (1, ))
    assert_size_stride(primals_1386, (128, ), (1, ))
    assert_size_stride(primals_1387, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1388, (128, ), (1, ))
    assert_size_stride(primals_1389, (128, ), (1, ))
    assert_size_stride(primals_1390, (128, ), (1, ))
    assert_size_stride(primals_1391, (128, ), (1, ))
    assert_size_stride(primals_1392, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1393, (128, ), (1, ))
    assert_size_stride(primals_1394, (128, ), (1, ))
    assert_size_stride(primals_1395, (128, ), (1, ))
    assert_size_stride(primals_1396, (128, ), (1, ))
    assert_size_stride(primals_1397, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1398, (128, ), (1, ))
    assert_size_stride(primals_1399, (128, ), (1, ))
    assert_size_stride(primals_1400, (128, ), (1, ))
    assert_size_stride(primals_1401, (128, ), (1, ))
    assert_size_stride(primals_1402, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_1403, (128, ), (1, ))
    assert_size_stride(primals_1404, (128, ), (1, ))
    assert_size_stride(primals_1405, (128, ), (1, ))
    assert_size_stride(primals_1406, (128, ), (1, ))
    assert_size_stride(primals_1407, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1408, (256, ), (1, ))
    assert_size_stride(primals_1409, (256, ), (1, ))
    assert_size_stride(primals_1410, (256, ), (1, ))
    assert_size_stride(primals_1411, (256, ), (1, ))
    assert_size_stride(primals_1412, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1413, (256, ), (1, ))
    assert_size_stride(primals_1414, (256, ), (1, ))
    assert_size_stride(primals_1415, (256, ), (1, ))
    assert_size_stride(primals_1416, (256, ), (1, ))
    assert_size_stride(primals_1417, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1418, (256, ), (1, ))
    assert_size_stride(primals_1419, (256, ), (1, ))
    assert_size_stride(primals_1420, (256, ), (1, ))
    assert_size_stride(primals_1421, (256, ), (1, ))
    assert_size_stride(primals_1422, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1423, (256, ), (1, ))
    assert_size_stride(primals_1424, (256, ), (1, ))
    assert_size_stride(primals_1425, (256, ), (1, ))
    assert_size_stride(primals_1426, (256, ), (1, ))
    assert_size_stride(primals_1427, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1428, (256, ), (1, ))
    assert_size_stride(primals_1429, (256, ), (1, ))
    assert_size_stride(primals_1430, (256, ), (1, ))
    assert_size_stride(primals_1431, (256, ), (1, ))
    assert_size_stride(primals_1432, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1433, (256, ), (1, ))
    assert_size_stride(primals_1434, (256, ), (1, ))
    assert_size_stride(primals_1435, (256, ), (1, ))
    assert_size_stride(primals_1436, (256, ), (1, ))
    assert_size_stride(primals_1437, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1438, (256, ), (1, ))
    assert_size_stride(primals_1439, (256, ), (1, ))
    assert_size_stride(primals_1440, (256, ), (1, ))
    assert_size_stride(primals_1441, (256, ), (1, ))
    assert_size_stride(primals_1442, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_1443, (256, ), (1, ))
    assert_size_stride(primals_1444, (256, ), (1, ))
    assert_size_stride(primals_1445, (256, ), (1, ))
    assert_size_stride(primals_1446, (256, ), (1, ))
    assert_size_stride(primals_1447, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_1448, (32, ), (1, ))
    assert_size_stride(primals_1449, (32, ), (1, ))
    assert_size_stride(primals_1450, (32, ), (1, ))
    assert_size_stride(primals_1451, (32, ), (1, ))
    assert_size_stride(primals_1452, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_1453, (32, ), (1, ))
    assert_size_stride(primals_1454, (32, ), (1, ))
    assert_size_stride(primals_1455, (32, ), (1, ))
    assert_size_stride(primals_1456, (32, ), (1, ))
    assert_size_stride(primals_1457, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_1458, (32, ), (1, ))
    assert_size_stride(primals_1459, (32, ), (1, ))
    assert_size_stride(primals_1460, (32, ), (1, ))
    assert_size_stride(primals_1461, (32, ), (1, ))
    assert_size_stride(primals_1462, (17, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_1463, (17, ), (1, ))
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
        assert_size_stride(buf30, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf31 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf30, primals_78, primals_79, primals_80, primals_81, buf31, 32768, grid=grid(32768), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf29, primals_82, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf33 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf32, primals_83, primals_84, primals_85, primals_86, buf33, 16384, grid=grid(16384), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf31, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf35 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf34, primals_88, primals_89, primals_90, primals_91, buf35, 32768, grid=grid(32768), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf37 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf36, primals_93, primals_94, primals_95, primals_96, buf31, buf37, 32768, grid=grid(32768), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf39 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_48, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf38, primals_98, primals_99, primals_100, primals_101, buf39, 32768, grid=grid(32768), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf41 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf40, primals_103, primals_104, primals_105, primals_106, buf37, buf41, 32768, grid=grid(32768), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf43 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf42, primals_108, primals_109, primals_110, primals_111, buf43, 32768, grid=grid(32768), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf45 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_58, out_59, out_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf44, primals_113, primals_114, primals_115, primals_116, buf41, buf45, 32768, grid=grid(32768), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf47 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf46, primals_118, primals_119, primals_120, primals_121, buf47, 32768, grid=grid(32768), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf33, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf51 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_69, out_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf50, primals_128, primals_129, primals_130, primals_131, buf51, 16384, grid=grid(16384), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf53 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_72, out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf52, primals_133, primals_134, primals_135, primals_136, buf33, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf55 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_76, out_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf54, primals_138, primals_139, primals_140, primals_141, buf55, 16384, grid=grid(16384), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [out_78], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf57 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_79, out_80, out_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf56, primals_143, primals_144, primals_145, primals_146, buf53, buf57, 16384, grid=grid(16384), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_82], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf59 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_83, out_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf58, primals_148, primals_149, primals_150, primals_151, buf59, 16384, grid=grid(16384), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf61 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_86, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf60, primals_153, primals_154, primals_155, primals_156, buf57, buf61, 16384, grid=grid(16384), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [out_89], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf63 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_90, out_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf62, primals_158, primals_159, primals_160, primals_161, buf63, 16384, grid=grid(16384), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf65 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_93, out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf64, primals_163, primals_164, primals_165, primals_166, buf61, buf65, 16384, grid=grid(16384), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf67 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_8.run(buf67, 16, grid=grid(16), stream=stream0)
        buf49 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_65, out_66, out_67, input_10, input_11, value, value_1, xi], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_9.run(buf48, primals_123, primals_124, primals_125, primals_126, buf45, buf67, buf66, primals_168, primals_169, primals_170, primals_171, buf49, buf68, 32768, grid=grid(32768), stream=stream0)
        del primals_126
        del primals_171
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf49, primals_172, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf70 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, value_2, value_3, xi_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf69, primals_173, primals_174, primals_175, primals_176, buf65, buf70, 16384, grid=grid(16384), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_177, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf72 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf71, primals_178, primals_179, primals_180, primals_181, buf72, 8192, grid=grid(8192), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf68, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf74 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf73, primals_183, primals_184, primals_185, primals_186, buf74, 32768, grid=grid(32768), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [out_99], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf76 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_100, out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf75, primals_188, primals_189, primals_190, primals_191, buf68, buf76, 32768, grid=grid(32768), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf78 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf77, primals_193, primals_194, primals_195, primals_196, buf78, 32768, grid=grid(32768), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf80 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf79, primals_198, primals_199, primals_200, primals_201, buf76, buf80, 32768, grid=grid(32768), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf82 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf81, primals_203, primals_204, primals_205, primals_206, buf82, 32768, grid=grid(32768), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf84 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf83, primals_208, primals_209, primals_210, primals_211, buf80, buf84, 32768, grid=grid(32768), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf86 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf85, primals_213, primals_214, primals_215, primals_216, buf86, 32768, grid=grid(32768), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf70, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf90 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf89, primals_223, primals_224, primals_225, primals_226, buf90, 16384, grid=grid(16384), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [out_127], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf92 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_128, out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf91, primals_228, primals_229, primals_230, primals_231, buf70, buf92, 16384, grid=grid(16384), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf94 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf93, primals_233, primals_234, primals_235, primals_236, buf94, 16384, grid=grid(16384), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf96 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_135, out_136, out_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf95, primals_238, primals_239, primals_240, primals_241, buf92, buf96, 16384, grid=grid(16384), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [out_138], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf98 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_139, out_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf97, primals_243, primals_244, primals_245, primals_246, buf98, 16384, grid=grid(16384), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf100 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_142, out_143, out_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf99, primals_248, primals_249, primals_250, primals_251, buf96, buf100, 16384, grid=grid(16384), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [out_145], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf102 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_146, out_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf101, primals_253, primals_254, primals_255, primals_256, buf102, 16384, grid=grid(16384), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf104 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_149, out_150, out_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf103, primals_258, primals_259, primals_260, primals_261, buf100, buf104, 16384, grid=grid(16384), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf104, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf72, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf106 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf105, primals_263, primals_264, primals_265, primals_266, buf106, 8192, grid=grid(8192), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [out_155], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf108 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_156, out_157, out_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf107, primals_268, primals_269, primals_270, primals_271, buf72, buf108, 8192, grid=grid(8192), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [out_159], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf110 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_160, out_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf109, primals_273, primals_274, primals_275, primals_276, buf110, 8192, grid=grid(8192), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [out_162], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf112 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_163, out_164, out_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf111, primals_278, primals_279, primals_280, primals_281, buf108, buf112, 8192, grid=grid(8192), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf114 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_167, out_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf113, primals_283, primals_284, primals_285, primals_286, buf114, 8192, grid=grid(8192), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [out_169], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf116 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_170, out_171, out_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf115, primals_288, primals_289, primals_290, primals_291, buf112, buf116, 8192, grid=grid(8192), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [out_173], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf118 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf117, primals_293, primals_294, primals_295, primals_296, buf118, 8192, grid=grid(8192), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf120 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_177, out_178, out_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf119, primals_298, primals_299, primals_300, primals_301, buf116, buf120, 8192, grid=grid(8192), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf120, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 32, 4, 4), (512, 16, 4, 1))
        buf123 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_11, input_22], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_13.run(buf123, 16, grid=grid(16), stream=stream0)
        buf88 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf124 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [out_121, out_122, out_123, input_18, input_19, input_21, input_22, value_4, value_5, value_6, xi_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14.run(buf125, buf87, primals_218, primals_219, primals_220, primals_221, buf84, buf67, buf121, primals_303, primals_304, primals_305, primals_306, buf123, buf122, primals_308, primals_309, primals_310, primals_311, buf88, 32768, grid=grid(32768), stream=stream0)
        del primals_221
        del primals_306
        del primals_311
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf88, primals_312, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf120, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf128 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_15.run(buf128, 8, grid=grid(8), stream=stream0)
        buf129 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_26, input_27, value_7, value_8, value_9, xi_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16.run(buf130, buf126, primals_313, primals_314, primals_315, primals_316, buf104, buf128, buf127, primals_318, primals_319, primals_320, primals_321, 16384, grid=grid(16384), stream=stream0)
        del primals_316
        del primals_321
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf88, primals_322, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf132 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf131, primals_323, primals_324, primals_325, primals_326, buf132, 8192, grid=grid(8192), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_327, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf104, primals_332, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf135 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_32, input_34, value_10, value_11, value_12, xi_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf136, buf133, primals_328, primals_329, primals_330, primals_331, buf134, primals_333, primals_334, primals_335, primals_336, buf120, 8192, grid=grid(8192), stream=stream0)
        del primals_331
        del primals_336
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf125, primals_337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf138 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_181, out_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf137, primals_338, primals_339, primals_340, primals_341, buf138, 32768, grid=grid(32768), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [out_183], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf140 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_184, out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf139, primals_343, primals_344, primals_345, primals_346, buf125, buf140, 32768, grid=grid(32768), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [out_187], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf142 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_188, out_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf141, primals_348, primals_349, primals_350, primals_351, buf142, 32768, grid=grid(32768), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf144 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_191, out_192, out_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf143, primals_353, primals_354, primals_355, primals_356, buf140, buf144, 32768, grid=grid(32768), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [out_194], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf146 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_195, out_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf145, primals_358, primals_359, primals_360, primals_361, buf146, 32768, grid=grid(32768), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [out_197], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf148 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_198, out_199, out_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf147, primals_363, primals_364, primals_365, primals_366, buf144, buf148, 32768, grid=grid(32768), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [out_201], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf150 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_202, out_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf149, primals_368, primals_369, primals_370, primals_371, buf150, 32768, grid=grid(32768), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf130, primals_377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf154 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf153, primals_378, primals_379, primals_380, primals_381, buf154, 16384, grid=grid(16384), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [out_211], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf156 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_212, out_213, out_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf155, primals_383, primals_384, primals_385, primals_386, buf130, buf156, 16384, grid=grid(16384), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [out_215], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf158 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_216, out_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf157, primals_388, primals_389, primals_390, primals_391, buf158, 16384, grid=grid(16384), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [out_218], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf160 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_219, out_220, out_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf159, primals_393, primals_394, primals_395, primals_396, buf156, buf160, 16384, grid=grid(16384), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [out_222], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf162 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_223, out_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf161, primals_398, primals_399, primals_400, primals_401, buf162, 16384, grid=grid(16384), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [out_225], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf164 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_226, out_227, out_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf163, primals_403, primals_404, primals_405, primals_406, buf160, buf164, 16384, grid=grid(16384), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [out_229], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf166 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_230, out_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf165, primals_408, primals_409, primals_410, primals_411, buf166, 16384, grid=grid(16384), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf168 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_233, out_234, out_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf167, primals_413, primals_414, primals_415, primals_416, buf164, buf168, 16384, grid=grid(16384), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf168, primals_457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf136, primals_417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf170 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_237, out_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf169, primals_418, primals_419, primals_420, primals_421, buf170, 8192, grid=grid(8192), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [out_239], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf172 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_240, out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf171, primals_423, primals_424, primals_425, primals_426, buf136, buf172, 8192, grid=grid(8192), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf174 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_244, out_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf173, primals_428, primals_429, primals_430, primals_431, buf174, 8192, grid=grid(8192), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [out_246], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf176 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_247, out_248, out_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf175, primals_433, primals_434, primals_435, primals_436, buf172, buf176, 8192, grid=grid(8192), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_437, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf178 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_251, out_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf177, primals_438, primals_439, primals_440, primals_441, buf178, 8192, grid=grid(8192), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [out_253], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf180 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_254, out_255, out_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf179, primals_443, primals_444, primals_445, primals_446, buf176, buf180, 8192, grid=grid(8192), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [out_257], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_447, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf182 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_258, out_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf181, primals_448, primals_449, primals_450, primals_451, buf182, 8192, grid=grid(8192), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf184 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_261, out_262, out_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf183, primals_453, primals_454, primals_455, primals_456, buf180, buf184, 8192, grid=grid(8192), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf184, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 32, 4, 4), (512, 16, 4, 1))
        buf152 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf187 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [out_205, out_206, out_207, input_36, input_37, input_39, input_40, value_13, value_14, value_15, xi_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14.run(buf188, buf151, primals_373, primals_374, primals_375, primals_376, buf148, buf67, buf185, primals_458, primals_459, primals_460, primals_461, buf123, buf186, primals_463, primals_464, primals_465, primals_466, buf152, 32768, grid=grid(32768), stream=stream0)
        del primals_376
        del primals_461
        del primals_466
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf152, primals_467, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf184, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf191 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [input_42, input_44, input_45, value_16, value_17, value_18, xi_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16.run(buf192, buf189, primals_468, primals_469, primals_470, primals_471, buf168, buf128, buf190, primals_473, primals_474, primals_475, primals_476, 16384, grid=grid(16384), stream=stream0)
        del primals_471
        del primals_476
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf152, primals_477, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf194 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf193, primals_478, primals_479, primals_480, primals_481, buf194, 8192, grid=grid(8192), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_482, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf168, primals_487, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf197 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_50, input_52, value_19, value_20, value_21, xi_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf198, buf195, primals_483, primals_484, primals_485, primals_486, buf196, primals_488, primals_489, primals_490, primals_491, buf184, 8192, grid=grid(8192), stream=stream0)
        del primals_486
        del primals_491
        # Topologically Sorted Source Nodes: [out_264], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf188, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf200 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf199, primals_493, primals_494, primals_495, primals_496, buf200, 32768, grid=grid(32768), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [out_267], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf202 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_268, out_269, out_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf201, primals_498, primals_499, primals_500, primals_501, buf188, buf202, 32768, grid=grid(32768), stream=stream0)
        del primals_501
        # Topologically Sorted Source Nodes: [out_271], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_502, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf204 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_272, out_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf203, primals_503, primals_504, primals_505, primals_506, buf204, 32768, grid=grid(32768), stream=stream0)
        del primals_506
        # Topologically Sorted Source Nodes: [out_274], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_507, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf206 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_275, out_276, out_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf205, primals_508, primals_509, primals_510, primals_511, buf202, buf206, 32768, grid=grid(32768), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [out_278], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf208 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_279, out_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf207, primals_513, primals_514, primals_515, primals_516, buf208, 32768, grid=grid(32768), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [out_281], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_517, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf210 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_282, out_283, out_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf209, primals_518, primals_519, primals_520, primals_521, buf206, buf210, 32768, grid=grid(32768), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [out_285], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf212 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_286, out_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf211, primals_523, primals_524, primals_525, primals_526, buf212, 32768, grid=grid(32768), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [out_288], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_527, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_292], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf192, primals_532, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf216 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_293, out_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf215, primals_533, primals_534, primals_535, primals_536, buf216, 16384, grid=grid(16384), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [out_295], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_537, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf218 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_296, out_297, out_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf217, primals_538, primals_539, primals_540, primals_541, buf192, buf218, 16384, grid=grid(16384), stream=stream0)
        del primals_541
        # Topologically Sorted Source Nodes: [out_299], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_542, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf220 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_300, out_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf219, primals_543, primals_544, primals_545, primals_546, buf220, 16384, grid=grid(16384), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [out_302], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_547, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf222 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_303, out_304, out_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf221, primals_548, primals_549, primals_550, primals_551, buf218, buf222, 16384, grid=grid(16384), stream=stream0)
        del primals_551
        # Topologically Sorted Source Nodes: [out_306], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf224 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_307, out_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf223, primals_553, primals_554, primals_555, primals_556, buf224, 16384, grid=grid(16384), stream=stream0)
        del primals_556
        # Topologically Sorted Source Nodes: [out_309], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_557, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf226 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_310, out_311, out_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf225, primals_558, primals_559, primals_560, primals_561, buf222, buf226, 16384, grid=grid(16384), stream=stream0)
        del primals_561
        # Topologically Sorted Source Nodes: [out_313], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_562, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf228 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_314, out_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf227, primals_563, primals_564, primals_565, primals_566, buf228, 16384, grid=grid(16384), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [out_316], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_567, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf230 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_317, out_318, out_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf229, primals_568, primals_569, primals_570, primals_571, buf226, buf230, 16384, grid=grid(16384), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf230, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf198, primals_572, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf232 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf231, primals_573, primals_574, primals_575, primals_576, buf232, 8192, grid=grid(8192), stream=stream0)
        del primals_576
        # Topologically Sorted Source Nodes: [out_323], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_577, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf234 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf233, primals_578, primals_579, primals_580, primals_581, buf198, buf234, 8192, grid=grid(8192), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [out_327], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_582, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf236 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_328, out_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf235, primals_583, primals_584, primals_585, primals_586, buf236, 8192, grid=grid(8192), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [out_330], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_587, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf238 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_331, out_332, out_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf237, primals_588, primals_589, primals_590, primals_591, buf234, buf238, 8192, grid=grid(8192), stream=stream0)
        del primals_591
        # Topologically Sorted Source Nodes: [out_334], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_592, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf240 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_335, out_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf239, primals_593, primals_594, primals_595, primals_596, buf240, 8192, grid=grid(8192), stream=stream0)
        del primals_596
        # Topologically Sorted Source Nodes: [out_337], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_597, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf242 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_338, out_339, out_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf241, primals_598, primals_599, primals_600, primals_601, buf238, buf242, 8192, grid=grid(8192), stream=stream0)
        del primals_601
        # Topologically Sorted Source Nodes: [out_341], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_602, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf244 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_342, out_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf243, primals_603, primals_604, primals_605, primals_606, buf244, 8192, grid=grid(8192), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [out_344], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_607, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf246 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_345, out_346, out_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf245, primals_608, primals_609, primals_610, primals_611, buf242, buf246, 8192, grid=grid(8192), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf246, primals_617, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 32, 4, 4), (512, 16, 4, 1))
        buf214 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf249 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf250 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [out_289, out_290, out_291, input_54, input_55, input_57, input_58, value_22, value_23, value_24, xi_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14.run(buf250, buf213, primals_528, primals_529, primals_530, primals_531, buf210, buf67, buf247, primals_613, primals_614, primals_615, primals_616, buf123, buf248, primals_618, primals_619, primals_620, primals_621, buf214, 32768, grid=grid(32768), stream=stream0)
        del primals_531
        del primals_616
        del primals_621
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf214, primals_622, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf246, primals_627, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf253 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [input_60, input_62, input_63, value_25, value_26, value_27, xi_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16.run(buf254, buf251, primals_623, primals_624, primals_625, primals_626, buf230, buf128, buf252, primals_628, primals_629, primals_630, primals_631, 16384, grid=grid(16384), stream=stream0)
        del primals_626
        del primals_631
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf214, primals_632, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf256 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf255, primals_633, primals_634, primals_635, primals_636, buf256, 8192, grid=grid(8192), stream=stream0)
        del primals_636
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_637, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf230, primals_642, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf259 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf260 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_70, value_28, value_29, value_30, xi_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf260, buf257, primals_638, primals_639, primals_640, primals_641, buf258, primals_643, primals_644, primals_645, primals_646, buf246, 8192, grid=grid(8192), stream=stream0)
        del primals_641
        del primals_646
        # Topologically Sorted Source Nodes: [out_348], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf250, primals_647, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf262 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_349, out_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf261, primals_648, primals_649, primals_650, primals_651, buf262, 32768, grid=grid(32768), stream=stream0)
        del primals_651
        # Topologically Sorted Source Nodes: [out_351], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_652, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf264 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_352, out_353, out_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf263, primals_653, primals_654, primals_655, primals_656, buf250, buf264, 32768, grid=grid(32768), stream=stream0)
        del primals_656
        # Topologically Sorted Source Nodes: [out_355], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_657, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf266 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_356, out_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf265, primals_658, primals_659, primals_660, primals_661, buf266, 32768, grid=grid(32768), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [out_358], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_662, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf268 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_359, out_360, out_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf267, primals_663, primals_664, primals_665, primals_666, buf264, buf268, 32768, grid=grid(32768), stream=stream0)
        del primals_666
        # Topologically Sorted Source Nodes: [out_362], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf270 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_363, out_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf269, primals_668, primals_669, primals_670, primals_671, buf270, 32768, grid=grid(32768), stream=stream0)
        del primals_671
        # Topologically Sorted Source Nodes: [out_365], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_672, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf272 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_366, out_367, out_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf271, primals_673, primals_674, primals_675, primals_676, buf268, buf272, 32768, grid=grid(32768), stream=stream0)
        del primals_676
        # Topologically Sorted Source Nodes: [out_369], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_677, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf274 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_370, out_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf273, primals_678, primals_679, primals_680, primals_681, buf274, 32768, grid=grid(32768), stream=stream0)
        del primals_681
        # Topologically Sorted Source Nodes: [out_372], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_682, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_376], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf254, primals_687, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf278 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_377, out_378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf277, primals_688, primals_689, primals_690, primals_691, buf278, 16384, grid=grid(16384), stream=stream0)
        del primals_691
        # Topologically Sorted Source Nodes: [out_379], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_692, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf280 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_380, out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf279, primals_693, primals_694, primals_695, primals_696, buf254, buf280, 16384, grid=grid(16384), stream=stream0)
        del primals_696
        # Topologically Sorted Source Nodes: [out_383], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_697, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf282 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_384, out_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf281, primals_698, primals_699, primals_700, primals_701, buf282, 16384, grid=grid(16384), stream=stream0)
        del primals_701
        # Topologically Sorted Source Nodes: [out_386], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_702, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf284 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_387, out_388, out_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf283, primals_703, primals_704, primals_705, primals_706, buf280, buf284, 16384, grid=grid(16384), stream=stream0)
        del primals_706
        # Topologically Sorted Source Nodes: [out_390], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_707, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf286 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_391, out_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf285, primals_708, primals_709, primals_710, primals_711, buf286, 16384, grid=grid(16384), stream=stream0)
        del primals_711
        # Topologically Sorted Source Nodes: [out_393], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_712, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf288 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_394, out_395, out_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf287, primals_713, primals_714, primals_715, primals_716, buf284, buf288, 16384, grid=grid(16384), stream=stream0)
        del primals_716
        # Topologically Sorted Source Nodes: [out_397], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_717, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf290 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_398, out_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf289, primals_718, primals_719, primals_720, primals_721, buf290, 16384, grid=grid(16384), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [out_400], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_722, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf292 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_401, out_402, out_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf291, primals_723, primals_724, primals_725, primals_726, buf288, buf292, 16384, grid=grid(16384), stream=stream0)
        del primals_726
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf292, primals_767, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_404], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf260, primals_727, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf294 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_405, out_406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf293, primals_728, primals_729, primals_730, primals_731, buf294, 8192, grid=grid(8192), stream=stream0)
        del primals_731
        # Topologically Sorted Source Nodes: [out_407], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_732, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf296 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_408, out_409, out_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf295, primals_733, primals_734, primals_735, primals_736, buf260, buf296, 8192, grid=grid(8192), stream=stream0)
        del primals_736
        # Topologically Sorted Source Nodes: [out_411], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_737, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf298 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_412, out_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf297, primals_738, primals_739, primals_740, primals_741, buf298, 8192, grid=grid(8192), stream=stream0)
        del primals_741
        # Topologically Sorted Source Nodes: [out_414], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_742, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf300 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_415, out_416, out_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf299, primals_743, primals_744, primals_745, primals_746, buf296, buf300, 8192, grid=grid(8192), stream=stream0)
        del primals_746
        # Topologically Sorted Source Nodes: [out_418], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_747, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf302 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_419, out_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf301, primals_748, primals_749, primals_750, primals_751, buf302, 8192, grid=grid(8192), stream=stream0)
        del primals_751
        # Topologically Sorted Source Nodes: [out_421], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_752, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf304 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_422, out_423, out_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf303, primals_753, primals_754, primals_755, primals_756, buf300, buf304, 8192, grid=grid(8192), stream=stream0)
        del primals_756
        # Topologically Sorted Source Nodes: [out_425], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_757, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf306 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_426, out_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf305, primals_758, primals_759, primals_760, primals_761, buf306, 8192, grid=grid(8192), stream=stream0)
        del primals_761
        # Topologically Sorted Source Nodes: [out_428], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_762, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf308 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_429, out_430, out_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf307, primals_763, primals_764, primals_765, primals_766, buf304, buf308, 8192, grid=grid(8192), stream=stream0)
        del primals_766
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf308, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 32, 4, 4), (512, 16, 4, 1))
        buf276 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf311 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf312 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [out_373, out_374, out_375, input_72, input_73, input_75, input_76, value_31, value_32, value_33, xi_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_14.run(buf312, buf275, primals_683, primals_684, primals_685, primals_686, buf272, buf67, buf309, primals_768, primals_769, primals_770, primals_771, buf123, buf310, primals_773, primals_774, primals_775, primals_776, buf276, 32768, grid=grid(32768), stream=stream0)
        del primals_686
        del primals_771
        del primals_776
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf276, primals_777, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf308, primals_782, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf315 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf316 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [input_78, input_80, input_81, value_34, value_35, value_36, xi_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_16.run(buf316, buf313, primals_778, primals_779, primals_780, primals_781, buf292, buf128, buf314, primals_783, primals_784, primals_785, primals_786, 16384, grid=grid(16384), stream=stream0)
        del primals_781
        del primals_786
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf276, primals_787, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf318 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, input_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf317, primals_788, primals_789, primals_790, primals_791, buf318, 8192, grid=grid(8192), stream=stream0)
        del primals_791
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_792, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf292, primals_797, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf321 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf322 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [input_86, input_88, value_37, value_38, value_39, xi_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf322, buf319, primals_793, primals_794, primals_795, primals_796, buf320, primals_798, primals_799, primals_800, primals_801, buf308, 8192, grid=grid(8192), stream=stream0)
        del primals_796
        del primals_801
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_802, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf324 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_90, input_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf323, primals_803, primals_804, primals_805, primals_806, buf324, 4096, grid=grid(4096), stream=stream0)
        del primals_806
        # Topologically Sorted Source Nodes: [out_432], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf312, primals_807, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf326 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_433, out_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf325, primals_808, primals_809, primals_810, primals_811, buf326, 32768, grid=grid(32768), stream=stream0)
        del primals_811
        # Topologically Sorted Source Nodes: [out_435], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_812, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf328 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_436, out_437, out_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf327, primals_813, primals_814, primals_815, primals_816, buf312, buf328, 32768, grid=grid(32768), stream=stream0)
        del primals_816
        # Topologically Sorted Source Nodes: [out_439], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_817, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf330 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_440, out_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf329, primals_818, primals_819, primals_820, primals_821, buf330, 32768, grid=grid(32768), stream=stream0)
        del primals_821
        # Topologically Sorted Source Nodes: [out_442], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_822, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf332 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_443, out_444, out_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf331, primals_823, primals_824, primals_825, primals_826, buf328, buf332, 32768, grid=grid(32768), stream=stream0)
        del primals_826
        # Topologically Sorted Source Nodes: [out_446], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_827, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf334 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_447, out_448], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf333, primals_828, primals_829, primals_830, primals_831, buf334, 32768, grid=grid(32768), stream=stream0)
        del primals_831
        # Topologically Sorted Source Nodes: [out_449], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_832, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf336 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_450, out_451, out_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf335, primals_833, primals_834, primals_835, primals_836, buf332, buf336, 32768, grid=grid(32768), stream=stream0)
        del primals_836
        # Topologically Sorted Source Nodes: [out_453], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_837, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf338 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_454, out_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf337, primals_838, primals_839, primals_840, primals_841, buf338, 32768, grid=grid(32768), stream=stream0)
        del primals_841
        # Topologically Sorted Source Nodes: [out_456], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_842, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf316, primals_847, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf342 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_461, out_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf341, primals_848, primals_849, primals_850, primals_851, buf342, 16384, grid=grid(16384), stream=stream0)
        del primals_851
        # Topologically Sorted Source Nodes: [out_463], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, primals_852, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf344 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_464, out_465, out_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf343, primals_853, primals_854, primals_855, primals_856, buf316, buf344, 16384, grid=grid(16384), stream=stream0)
        del primals_856
        # Topologically Sorted Source Nodes: [out_467], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, primals_857, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf346 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_468, out_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf345, primals_858, primals_859, primals_860, primals_861, buf346, 16384, grid=grid(16384), stream=stream0)
        del primals_861
        # Topologically Sorted Source Nodes: [out_470], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_862, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf348 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_471, out_472, out_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf347, primals_863, primals_864, primals_865, primals_866, buf344, buf348, 16384, grid=grid(16384), stream=stream0)
        del primals_866
        # Topologically Sorted Source Nodes: [out_474], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, primals_867, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf350 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_475, out_476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf349, primals_868, primals_869, primals_870, primals_871, buf350, 16384, grid=grid(16384), stream=stream0)
        del primals_871
        # Topologically Sorted Source Nodes: [out_477], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_872, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf352 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_478, out_479, out_480], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf351, primals_873, primals_874, primals_875, primals_876, buf348, buf352, 16384, grid=grid(16384), stream=stream0)
        del primals_876
        # Topologically Sorted Source Nodes: [out_481], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_877, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf354 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_482, out_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf353, primals_878, primals_879, primals_880, primals_881, buf354, 16384, grid=grid(16384), stream=stream0)
        del primals_881
        # Topologically Sorted Source Nodes: [out_484], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, primals_882, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf356 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_485, out_486, out_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf355, primals_883, primals_884, primals_885, primals_886, buf352, buf356, 16384, grid=grid(16384), stream=stream0)
        del primals_886
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf356, primals_967, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_488], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf322, primals_887, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf358 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_489, out_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf357, primals_888, primals_889, primals_890, primals_891, buf358, 8192, grid=grid(8192), stream=stream0)
        del primals_891
        # Topologically Sorted Source Nodes: [out_491], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_892, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf360 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_492, out_493, out_494], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf359, primals_893, primals_894, primals_895, primals_896, buf322, buf360, 8192, grid=grid(8192), stream=stream0)
        del primals_896
        # Topologically Sorted Source Nodes: [out_495], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_897, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf362 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_496, out_497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf361, primals_898, primals_899, primals_900, primals_901, buf362, 8192, grid=grid(8192), stream=stream0)
        del primals_901
        # Topologically Sorted Source Nodes: [out_498], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_902, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf364 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_499, out_500, out_501], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf363, primals_903, primals_904, primals_905, primals_906, buf360, buf364, 8192, grid=grid(8192), stream=stream0)
        del primals_906
        # Topologically Sorted Source Nodes: [out_502], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_907, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf366 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_503, out_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf365, primals_908, primals_909, primals_910, primals_911, buf366, 8192, grid=grid(8192), stream=stream0)
        del primals_911
        # Topologically Sorted Source Nodes: [out_505], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_912, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf368 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_506, out_507, out_508], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf367, primals_913, primals_914, primals_915, primals_916, buf364, buf368, 8192, grid=grid(8192), stream=stream0)
        del primals_916
        # Topologically Sorted Source Nodes: [out_509], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_917, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf370 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_510, out_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf369, primals_918, primals_919, primals_920, primals_921, buf370, 8192, grid=grid(8192), stream=stream0)
        del primals_921
        # Topologically Sorted Source Nodes: [out_512], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_922, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf372 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_513, out_514, out_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf371, primals_923, primals_924, primals_925, primals_926, buf368, buf372, 8192, grid=grid(8192), stream=stream0)
        del primals_926
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf372, primals_972, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 32, 4, 4), (512, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out_516], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf324, primals_927, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf374 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_517, out_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf373, primals_928, primals_929, primals_930, primals_931, buf374, 4096, grid=grid(4096), stream=stream0)
        del primals_931
        # Topologically Sorted Source Nodes: [out_519], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_932, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf376 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_520, out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf375, primals_933, primals_934, primals_935, primals_936, buf324, buf376, 4096, grid=grid(4096), stream=stream0)
        del primals_936
        # Topologically Sorted Source Nodes: [out_523], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_937, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf378 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_524, out_525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf377, primals_938, primals_939, primals_940, primals_941, buf378, 4096, grid=grid(4096), stream=stream0)
        del primals_941
        # Topologically Sorted Source Nodes: [out_526], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_942, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf380 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_527, out_528, out_529], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf379, primals_943, primals_944, primals_945, primals_946, buf376, buf380, 4096, grid=grid(4096), stream=stream0)
        del primals_946
        # Topologically Sorted Source Nodes: [out_530], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_947, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf382 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_531, out_532], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf381, primals_948, primals_949, primals_950, primals_951, buf382, 4096, grid=grid(4096), stream=stream0)
        del primals_951
        # Topologically Sorted Source Nodes: [out_533], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_952, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf384 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_534, out_535, out_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf383, primals_953, primals_954, primals_955, primals_956, buf380, buf384, 4096, grid=grid(4096), stream=stream0)
        del primals_956
        # Topologically Sorted Source Nodes: [out_537], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_957, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf386 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_538, out_539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf385, primals_958, primals_959, primals_960, primals_961, buf386, 4096, grid=grid(4096), stream=stream0)
        del primals_961
        # Topologically Sorted Source Nodes: [out_540], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, primals_962, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf388 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_541, out_542, out_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf387, primals_963, primals_964, primals_965, primals_966, buf384, buf388, 4096, grid=grid(4096), stream=stream0)
        del primals_966
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf388, primals_977, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (4, 32, 2, 2), (128, 4, 2, 1))
        buf392 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_11, input_100], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_21.run(buf392, 16, grid=grid(16), stream=stream0)
        buf340 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf393 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf394 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [out_457, out_458, out_459, input_93, input_94, input_96, input_97, input_99, input_100, value_40, value_41, value_42, value_43, xi_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22.run(buf394, buf339, primals_843, primals_844, primals_845, primals_846, buf336, buf67, buf389, primals_968, primals_969, primals_970, primals_971, buf123, buf390, primals_973, primals_974, primals_975, primals_976, buf392, buf391, primals_978, primals_979, primals_980, primals_981, buf340, 32768, grid=grid(32768), stream=stream0)
        del primals_846
        del primals_971
        del primals_976
        del primals_981
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf340, primals_982, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf372, primals_987, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf388, primals_992, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 64, 2, 2), (256, 4, 2, 1))
        buf398 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_27, input_108], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_23.run(buf398, 8, grid=grid(8), stream=stream0)
        buf399 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf400 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [input_102, input_104, input_105, input_107, input_108, value_44, value_45, value_46, value_47, xi_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24.run(buf400, buf395, primals_983, primals_984, primals_985, primals_986, buf356, buf128, buf396, primals_988, primals_989, primals_990, primals_991, buf398, buf397, primals_993, primals_994, primals_995, primals_996, 16384, grid=grid(16384), stream=stream0)
        del primals_986
        del primals_991
        del primals_996
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf340, primals_997, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf402 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf401, primals_998, primals_999, primals_1000, primals_1001, buf402, 8192, grid=grid(8192), stream=stream0)
        del primals_1001
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_1002, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf356, primals_1007, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf388, primals_1012, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (4, 128, 2, 2), (512, 4, 2, 1))
        buf406 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_25.run(buf406, 4, grid=grid(4), stream=stream0)
        buf407 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf408 = buf407; del buf407  # reuse
        buf409 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [input_113, input_115, input_117, input_118, value_48, value_49, value_50, value_51, xi_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf409, buf403, primals_1003, primals_1004, primals_1005, primals_1006, buf404, primals_1008, primals_1009, primals_1010, primals_1011, buf372, buf406, buf405, primals_1013, primals_1014, primals_1015, primals_1016, 8192, grid=grid(8192), stream=stream0)
        del primals_1006
        del primals_1011
        del primals_1016
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf340, primals_1017, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf411 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_120, input_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf410, primals_1018, primals_1019, primals_1020, primals_1021, buf411, 8192, grid=grid(8192), stream=stream0)
        del primals_1021
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_1022, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 32, 4, 4), (512, 16, 4, 1))
        buf413 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_123, input_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf412, primals_1023, primals_1024, primals_1025, primals_1026, buf413, 2048, grid=grid(2048), stream=stream0)
        del primals_1026
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_1027, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf356, primals_1032, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf416 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf415, primals_1033, primals_1034, primals_1035, primals_1036, buf416, 4096, grid=grid(4096), stream=stream0)
        del primals_1036
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_1037, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_132], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf372, primals_1042, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf419 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        buf420 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [input_126, input_131, input_133, value_52, value_53, value_54, value_55, xi_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf420, buf414, primals_1028, primals_1029, primals_1030, primals_1031, buf417, primals_1038, primals_1039, primals_1040, primals_1041, buf418, primals_1043, primals_1044, primals_1045, primals_1046, buf388, 4096, grid=grid(4096), stream=stream0)
        del primals_1031
        del primals_1041
        del primals_1046
        # Topologically Sorted Source Nodes: [out_544], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf394, primals_1047, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf422 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_545, out_546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf421, primals_1048, primals_1049, primals_1050, primals_1051, buf422, 32768, grid=grid(32768), stream=stream0)
        del primals_1051
        # Topologically Sorted Source Nodes: [out_547], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_1052, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf424 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_548, out_549, out_550], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf423, primals_1053, primals_1054, primals_1055, primals_1056, buf394, buf424, 32768, grid=grid(32768), stream=stream0)
        del primals_1056
        # Topologically Sorted Source Nodes: [out_551], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_1057, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf426 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_552, out_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf425, primals_1058, primals_1059, primals_1060, primals_1061, buf426, 32768, grid=grid(32768), stream=stream0)
        del primals_1061
        # Topologically Sorted Source Nodes: [out_554], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_1062, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf428 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_555, out_556, out_557], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf427, primals_1063, primals_1064, primals_1065, primals_1066, buf424, buf428, 32768, grid=grid(32768), stream=stream0)
        del primals_1066
        # Topologically Sorted Source Nodes: [out_558], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_1067, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf430 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_559, out_560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf429, primals_1068, primals_1069, primals_1070, primals_1071, buf430, 32768, grid=grid(32768), stream=stream0)
        del primals_1071
        # Topologically Sorted Source Nodes: [out_561], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_1072, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf432 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_562, out_563, out_564], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf431, primals_1073, primals_1074, primals_1075, primals_1076, buf428, buf432, 32768, grid=grid(32768), stream=stream0)
        del primals_1076
        # Topologically Sorted Source Nodes: [out_565], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_1077, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf434 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_566, out_567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf433, primals_1078, primals_1079, primals_1080, primals_1081, buf434, 32768, grid=grid(32768), stream=stream0)
        del primals_1081
        # Topologically Sorted Source Nodes: [out_568], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, primals_1082, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_572], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf400, primals_1087, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf438 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_573, out_574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf437, primals_1088, primals_1089, primals_1090, primals_1091, buf438, 16384, grid=grid(16384), stream=stream0)
        del primals_1091
        # Topologically Sorted Source Nodes: [out_575], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_1092, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf440 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_576, out_577, out_578], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf439, primals_1093, primals_1094, primals_1095, primals_1096, buf400, buf440, 16384, grid=grid(16384), stream=stream0)
        del primals_1096
        # Topologically Sorted Source Nodes: [out_579], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_1097, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf442 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_580, out_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf441, primals_1098, primals_1099, primals_1100, primals_1101, buf442, 16384, grid=grid(16384), stream=stream0)
        del primals_1101
        # Topologically Sorted Source Nodes: [out_582], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_1102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf444 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_583, out_584, out_585], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf443, primals_1103, primals_1104, primals_1105, primals_1106, buf440, buf444, 16384, grid=grid(16384), stream=stream0)
        del primals_1106
        # Topologically Sorted Source Nodes: [out_586], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_1107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf446 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_587, out_588], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf445, primals_1108, primals_1109, primals_1110, primals_1111, buf446, 16384, grid=grid(16384), stream=stream0)
        del primals_1111
        # Topologically Sorted Source Nodes: [out_589], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_1112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf448 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_590, out_591, out_592], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf447, primals_1113, primals_1114, primals_1115, primals_1116, buf444, buf448, 16384, grid=grid(16384), stream=stream0)
        del primals_1116
        # Topologically Sorted Source Nodes: [out_593], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_1117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf450 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_594, out_595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf449, primals_1118, primals_1119, primals_1120, primals_1121, buf450, 16384, grid=grid(16384), stream=stream0)
        del primals_1121
        # Topologically Sorted Source Nodes: [out_596], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_1122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf452 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_597, out_598, out_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf451, primals_1123, primals_1124, primals_1125, primals_1126, buf448, buf452, 16384, grid=grid(16384), stream=stream0)
        del primals_1126
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf452, primals_1207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_600], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf409, primals_1127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf454 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_601, out_602], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf453, primals_1128, primals_1129, primals_1130, primals_1131, buf454, 8192, grid=grid(8192), stream=stream0)
        del primals_1131
        # Topologically Sorted Source Nodes: [out_603], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, primals_1132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf456 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_604, out_605, out_606], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf455, primals_1133, primals_1134, primals_1135, primals_1136, buf409, buf456, 8192, grid=grid(8192), stream=stream0)
        del primals_1136
        # Topologically Sorted Source Nodes: [out_607], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf456, primals_1137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf458 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_608, out_609], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf457, primals_1138, primals_1139, primals_1140, primals_1141, buf458, 8192, grid=grid(8192), stream=stream0)
        del primals_1141
        # Topologically Sorted Source Nodes: [out_610], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_1142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf460 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_611, out_612, out_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf459, primals_1143, primals_1144, primals_1145, primals_1146, buf456, buf460, 8192, grid=grid(8192), stream=stream0)
        del primals_1146
        # Topologically Sorted Source Nodes: [out_614], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_1147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf462 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_615, out_616], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf461, primals_1148, primals_1149, primals_1150, primals_1151, buf462, 8192, grid=grid(8192), stream=stream0)
        del primals_1151
        # Topologically Sorted Source Nodes: [out_617], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_1152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf464 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_618, out_619, out_620], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf463, primals_1153, primals_1154, primals_1155, primals_1156, buf460, buf464, 8192, grid=grid(8192), stream=stream0)
        del primals_1156
        # Topologically Sorted Source Nodes: [out_621], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, primals_1157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf466 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_622, out_623], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf465, primals_1158, primals_1159, primals_1160, primals_1161, buf466, 8192, grid=grid(8192), stream=stream0)
        del primals_1161
        # Topologically Sorted Source Nodes: [out_624], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_1162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf468 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_625, out_626, out_627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf467, primals_1163, primals_1164, primals_1165, primals_1166, buf464, buf468, 8192, grid=grid(8192), stream=stream0)
        del primals_1166
        # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf468, primals_1212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 32, 4, 4), (512, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out_628], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf420, primals_1167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf470 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_629, out_630], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf469, primals_1168, primals_1169, primals_1170, primals_1171, buf470, 4096, grid=grid(4096), stream=stream0)
        del primals_1171
        # Topologically Sorted Source Nodes: [out_631], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_1172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf472 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_632, out_633, out_634], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf471, primals_1173, primals_1174, primals_1175, primals_1176, buf420, buf472, 4096, grid=grid(4096), stream=stream0)
        del primals_1176
        # Topologically Sorted Source Nodes: [out_635], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_1177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf474 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_636, out_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf473, primals_1178, primals_1179, primals_1180, primals_1181, buf474, 4096, grid=grid(4096), stream=stream0)
        del primals_1181
        # Topologically Sorted Source Nodes: [out_638], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_1182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf476 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_639, out_640, out_641], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf475, primals_1183, primals_1184, primals_1185, primals_1186, buf472, buf476, 4096, grid=grid(4096), stream=stream0)
        del primals_1186
        # Topologically Sorted Source Nodes: [out_642], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_1187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf478 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_643, out_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf477, primals_1188, primals_1189, primals_1190, primals_1191, buf478, 4096, grid=grid(4096), stream=stream0)
        del primals_1191
        # Topologically Sorted Source Nodes: [out_645], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_1192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf480 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_646, out_647, out_648], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf479, primals_1193, primals_1194, primals_1195, primals_1196, buf476, buf480, 4096, grid=grid(4096), stream=stream0)
        del primals_1196
        # Topologically Sorted Source Nodes: [out_649], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_1197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf482 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_650, out_651], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf481, primals_1198, primals_1199, primals_1200, primals_1201, buf482, 4096, grid=grid(4096), stream=stream0)
        del primals_1201
        # Topologically Sorted Source Nodes: [out_652], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_1202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf484 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_653, out_654, out_655], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf483, primals_1203, primals_1204, primals_1205, primals_1206, buf480, buf484, 4096, grid=grid(4096), stream=stream0)
        del primals_1206
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf484, primals_1217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 32, 2, 2), (128, 4, 2, 1))
        buf436 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf488 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf489 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [out_569, out_570, out_571, input_135, input_136, input_138, input_139, input_141, input_142, value_56, value_57, value_58, value_59, xi_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22.run(buf489, buf435, primals_1083, primals_1084, primals_1085, primals_1086, buf432, buf67, buf485, primals_1208, primals_1209, primals_1210, primals_1211, buf123, buf486, primals_1213, primals_1214, primals_1215, primals_1216, buf392, buf487, primals_1218, primals_1219, primals_1220, primals_1221, buf436, 32768, grid=grid(32768), stream=stream0)
        del primals_1086
        del primals_1211
        del primals_1216
        del primals_1221
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf436, primals_1222, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf468, primals_1227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf484, primals_1232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (4, 64, 2, 2), (256, 4, 2, 1))
        buf493 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf494 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [input_144, input_146, input_147, input_149, input_150, value_60, value_61, value_62, value_63, xi_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_24.run(buf494, buf490, primals_1223, primals_1224, primals_1225, primals_1226, buf452, buf128, buf491, primals_1228, primals_1229, primals_1230, primals_1231, buf398, buf492, primals_1233, primals_1234, primals_1235, primals_1236, 16384, grid=grid(16384), stream=stream0)
        del primals_1226
        del primals_1231
        del primals_1236
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf436, primals_1237, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf496 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_152, input_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf495, primals_1238, primals_1239, primals_1240, primals_1241, buf496, 8192, grid=grid(8192), stream=stream0)
        del primals_1241
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_1242, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf452, primals_1247, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 128, 4, 4), (2048, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_158], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf484, primals_1252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 128, 2, 2), (512, 4, 2, 1))
        buf500 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf501 = buf500; del buf500  # reuse
        buf502 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [input_155, input_157, input_159, input_160, value_64, value_65, value_66, value_67, xi_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_26.run(buf502, buf497, primals_1243, primals_1244, primals_1245, primals_1246, buf498, primals_1248, primals_1249, primals_1250, primals_1251, buf468, buf406, buf499, primals_1253, primals_1254, primals_1255, primals_1256, 8192, grid=grid(8192), stream=stream0)
        del primals_1246
        del primals_1251
        del primals_1256
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf436, primals_1257, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf504 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf503, primals_1258, primals_1259, primals_1260, primals_1261, buf504, 8192, grid=grid(8192), stream=stream0)
        del primals_1261
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_1262, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (4, 32, 4, 4), (512, 16, 4, 1))
        buf506 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_165, input_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf505, primals_1263, primals_1264, primals_1265, primals_1266, buf506, 2048, grid=grid(2048), stream=stream0)
        del primals_1266
        # Topologically Sorted Source Nodes: [input_167], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_1267, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf452, primals_1272, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf509 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf508, primals_1273, primals_1274, primals_1275, primals_1276, buf509, 4096, grid=grid(4096), stream=stream0)
        del primals_1276
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_1277, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf468, primals_1282, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf512 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        buf513 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [input_168, input_173, input_175, value_68, value_69, value_70, value_71, xi_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf513, buf507, primals_1268, primals_1269, primals_1270, primals_1271, buf510, primals_1278, primals_1279, primals_1280, primals_1281, buf511, primals_1283, primals_1284, primals_1285, primals_1286, buf484, 4096, grid=grid(4096), stream=stream0)
        del primals_1271
        del primals_1281
        del primals_1286
        # Topologically Sorted Source Nodes: [out_656], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf489, primals_1287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf515 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_657, out_658], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf514, primals_1288, primals_1289, primals_1290, primals_1291, buf515, 32768, grid=grid(32768), stream=stream0)
        del primals_1291
        # Topologically Sorted Source Nodes: [out_659], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_1292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf517 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_660, out_661, out_662], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf516, primals_1293, primals_1294, primals_1295, primals_1296, buf489, buf517, 32768, grid=grid(32768), stream=stream0)
        del primals_1296
        # Topologically Sorted Source Nodes: [out_663], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_1297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf519 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_664, out_665], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf518, primals_1298, primals_1299, primals_1300, primals_1301, buf519, 32768, grid=grid(32768), stream=stream0)
        del primals_1301
        # Topologically Sorted Source Nodes: [out_666], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_1302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf521 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_667, out_668, out_669], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf520, primals_1303, primals_1304, primals_1305, primals_1306, buf517, buf521, 32768, grid=grid(32768), stream=stream0)
        del primals_1306
        # Topologically Sorted Source Nodes: [out_670], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_1307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf523 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_671, out_672], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf522, primals_1308, primals_1309, primals_1310, primals_1311, buf523, 32768, grid=grid(32768), stream=stream0)
        del primals_1311
        # Topologically Sorted Source Nodes: [out_673], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_1312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf525 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_674, out_675, out_676], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf524, primals_1313, primals_1314, primals_1315, primals_1316, buf521, buf525, 32768, grid=grid(32768), stream=stream0)
        del primals_1316
        # Topologically Sorted Source Nodes: [out_677], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_1317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf527 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_678, out_679], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf526, primals_1318, primals_1319, primals_1320, primals_1321, buf527, 32768, grid=grid(32768), stream=stream0)
        del primals_1321
        # Topologically Sorted Source Nodes: [out_680], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_1322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [out_684], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf494, primals_1327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf531 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_685, out_686], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf530, primals_1328, primals_1329, primals_1330, primals_1331, buf531, 16384, grid=grid(16384), stream=stream0)
        del primals_1331
        # Topologically Sorted Source Nodes: [out_687], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_1332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf533 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_688, out_689, out_690], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf532, primals_1333, primals_1334, primals_1335, primals_1336, buf494, buf533, 16384, grid=grid(16384), stream=stream0)
        del primals_1336
        # Topologically Sorted Source Nodes: [out_691], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_1337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf535 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_692, out_693], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf534, primals_1338, primals_1339, primals_1340, primals_1341, buf535, 16384, grid=grid(16384), stream=stream0)
        del primals_1341
        # Topologically Sorted Source Nodes: [out_694], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_1342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf537 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_695, out_696, out_697], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf536, primals_1343, primals_1344, primals_1345, primals_1346, buf533, buf537, 16384, grid=grid(16384), stream=stream0)
        del primals_1346
        # Topologically Sorted Source Nodes: [out_698], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_1347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf539 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_699, out_700], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf538, primals_1348, primals_1349, primals_1350, primals_1351, buf539, 16384, grid=grid(16384), stream=stream0)
        del primals_1351
        # Topologically Sorted Source Nodes: [out_701], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_1352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf541 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_702, out_703, out_704], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf540, primals_1353, primals_1354, primals_1355, primals_1356, buf537, buf541, 16384, grid=grid(16384), stream=stream0)
        del primals_1356
        # Topologically Sorted Source Nodes: [out_705], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_1357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf543 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_706, out_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf542, primals_1358, primals_1359, primals_1360, primals_1361, buf543, 16384, grid=grid(16384), stream=stream0)
        del primals_1361
        # Topologically Sorted Source Nodes: [out_708], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_1362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf545 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_709, out_710, out_711], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf544, primals_1363, primals_1364, primals_1365, primals_1366, buf541, buf545, 16384, grid=grid(16384), stream=stream0)
        del primals_1366
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(buf545, primals_1447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [out_712], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf502, primals_1367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf547 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_713, out_714], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf546, primals_1368, primals_1369, primals_1370, primals_1371, buf547, 8192, grid=grid(8192), stream=stream0)
        del primals_1371
        # Topologically Sorted Source Nodes: [out_715], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_1372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf549 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_716, out_717, out_718], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf548, primals_1373, primals_1374, primals_1375, primals_1376, buf502, buf549, 8192, grid=grid(8192), stream=stream0)
        del primals_1376
        # Topologically Sorted Source Nodes: [out_719], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, primals_1377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf551 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_720, out_721], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf550, primals_1378, primals_1379, primals_1380, primals_1381, buf551, 8192, grid=grid(8192), stream=stream0)
        del primals_1381
        # Topologically Sorted Source Nodes: [out_722], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_1382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf553 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_723, out_724, out_725], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf552, primals_1383, primals_1384, primals_1385, primals_1386, buf549, buf553, 8192, grid=grid(8192), stream=stream0)
        del primals_1386
        # Topologically Sorted Source Nodes: [out_726], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, primals_1387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf555 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_727, out_728], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf554, primals_1388, primals_1389, primals_1390, primals_1391, buf555, 8192, grid=grid(8192), stream=stream0)
        del primals_1391
        # Topologically Sorted Source Nodes: [out_729], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf555, primals_1392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf557 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_730, out_731, out_732], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf556, primals_1393, primals_1394, primals_1395, primals_1396, buf553, buf557, 8192, grid=grid(8192), stream=stream0)
        del primals_1396
        # Topologically Sorted Source Nodes: [out_733], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_1397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf559 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_734, out_735], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf558, primals_1398, primals_1399, primals_1400, primals_1401, buf559, 8192, grid=grid(8192), stream=stream0)
        del primals_1401
        # Topologically Sorted Source Nodes: [out_736], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_1402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf561 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_737, out_738, out_739], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf560, primals_1403, primals_1404, primals_1405, primals_1406, buf557, buf561, 8192, grid=grid(8192), stream=stream0)
        del primals_1406
        # Topologically Sorted Source Nodes: [input_179], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf561, primals_1452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 32, 4, 4), (512, 16, 4, 1))
        # Topologically Sorted Source Nodes: [out_740], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf513, primals_1407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf563 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_741, out_742], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf562, primals_1408, primals_1409, primals_1410, primals_1411, buf563, 4096, grid=grid(4096), stream=stream0)
        del primals_1411
        # Topologically Sorted Source Nodes: [out_743], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, primals_1412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf565 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_744, out_745, out_746], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf564, primals_1413, primals_1414, primals_1415, primals_1416, buf513, buf565, 4096, grid=grid(4096), stream=stream0)
        del primals_1416
        # Topologically Sorted Source Nodes: [out_747], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_1417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf567 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_748, out_749], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf566, primals_1418, primals_1419, primals_1420, primals_1421, buf567, 4096, grid=grid(4096), stream=stream0)
        del primals_1421
        # Topologically Sorted Source Nodes: [out_750], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, primals_1422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf569 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_751, out_752, out_753], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf568, primals_1423, primals_1424, primals_1425, primals_1426, buf565, buf569, 4096, grid=grid(4096), stream=stream0)
        del primals_1426
        # Topologically Sorted Source Nodes: [out_754], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_1427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf571 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_755, out_756], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf570, primals_1428, primals_1429, primals_1430, primals_1431, buf571, 4096, grid=grid(4096), stream=stream0)
        del primals_1431
        # Topologically Sorted Source Nodes: [out_757], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_1432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf573 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_758, out_759, out_760], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf572, primals_1433, primals_1434, primals_1435, primals_1436, buf569, buf573, 4096, grid=grid(4096), stream=stream0)
        del primals_1436
        # Topologically Sorted Source Nodes: [out_761], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_1437, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf575 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_762, out_763], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf574, primals_1438, primals_1439, primals_1440, primals_1441, buf575, 4096, grid=grid(4096), stream=stream0)
        del primals_1441
        # Topologically Sorted Source Nodes: [out_764], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_1442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf577 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_765, out_766, out_767], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf576, primals_1443, primals_1444, primals_1445, primals_1446, buf573, buf577, 4096, grid=grid(4096), stream=stream0)
        del primals_1446
        # Topologically Sorted Source Nodes: [input_182], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf577, primals_1457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (4, 32, 2, 2), (128, 4, 2, 1))
        buf581 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf582 = buf581; del buf581  # reuse
        buf585 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_681, out_682, out_683, input_177, input_178, input_180, input_181, input_183, input_184, value_72, value_73, value_74, value_75, relu_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten._unsafe_index, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_threshold_backward_30.run(buf582, buf528, primals_1323, primals_1324, primals_1325, primals_1326, buf525, buf67, buf578, primals_1448, primals_1449, primals_1450, primals_1451, buf123, buf579, primals_1453, primals_1454, primals_1455, primals_1456, buf392, buf580, primals_1458, primals_1459, primals_1460, primals_1461, buf585, 32768, grid=grid(32768), stream=stream0)
        del primals_1326
        del primals_1451
        del primals_1456
        del primals_1461
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, primals_1462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (4, 17, 16, 16), (4352, 256, 16, 1))
        buf584 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf584, primals_1463, 17408, grid=grid(17408), stream=stream0)
        del primals_1463
    return (buf584, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_465, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_525, primals_527, primals_528, primals_529, primals_530, primals_532, primals_533, primals_534, primals_535, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_562, primals_563, primals_564, primals_565, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_595, primals_597, primals_598, primals_599, primals_600, primals_602, primals_603, primals_604, primals_605, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_627, primals_628, primals_629, primals_630, primals_632, primals_633, primals_634, primals_635, primals_637, primals_638, primals_639, primals_640, primals_642, primals_643, primals_644, primals_645, primals_647, primals_648, primals_649, primals_650, primals_652, primals_653, primals_654, primals_655, primals_657, primals_658, primals_659, primals_660, primals_662, primals_663, primals_664, primals_665, primals_667, primals_668, primals_669, primals_670, primals_672, primals_673, primals_674, primals_675, primals_677, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_685, primals_687, primals_688, primals_689, primals_690, primals_692, primals_693, primals_694, primals_695, primals_697, primals_698, primals_699, primals_700, primals_702, primals_703, primals_704, primals_705, primals_707, primals_708, primals_709, primals_710, primals_712, primals_713, primals_714, primals_715, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_735, primals_737, primals_738, primals_739, primals_740, primals_742, primals_743, primals_744, primals_745, primals_747, primals_748, primals_749, primals_750, primals_752, primals_753, primals_754, primals_755, primals_757, primals_758, primals_759, primals_760, primals_762, primals_763, primals_764, primals_765, primals_767, primals_768, primals_769, primals_770, primals_772, primals_773, primals_774, primals_775, primals_777, primals_778, primals_779, primals_780, primals_782, primals_783, primals_784, primals_785, primals_787, primals_788, primals_789, primals_790, primals_792, primals_793, primals_794, primals_795, primals_797, primals_798, primals_799, primals_800, primals_802, primals_803, primals_804, primals_805, primals_807, primals_808, primals_809, primals_810, primals_812, primals_813, primals_814, primals_815, primals_817, primals_818, primals_819, primals_820, primals_822, primals_823, primals_824, primals_825, primals_827, primals_828, primals_829, primals_830, primals_832, primals_833, primals_834, primals_835, primals_837, primals_838, primals_839, primals_840, primals_842, primals_843, primals_844, primals_845, primals_847, primals_848, primals_849, primals_850, primals_852, primals_853, primals_854, primals_855, primals_857, primals_858, primals_859, primals_860, primals_862, primals_863, primals_864, primals_865, primals_867, primals_868, primals_869, primals_870, primals_872, primals_873, primals_874, primals_875, primals_877, primals_878, primals_879, primals_880, primals_882, primals_883, primals_884, primals_885, primals_887, primals_888, primals_889, primals_890, primals_892, primals_893, primals_894, primals_895, primals_897, primals_898, primals_899, primals_900, primals_902, primals_903, primals_904, primals_905, primals_907, primals_908, primals_909, primals_910, primals_912, primals_913, primals_914, primals_915, primals_917, primals_918, primals_919, primals_920, primals_922, primals_923, primals_924, primals_925, primals_927, primals_928, primals_929, primals_930, primals_932, primals_933, primals_934, primals_935, primals_937, primals_938, primals_939, primals_940, primals_942, primals_943, primals_944, primals_945, primals_947, primals_948, primals_949, primals_950, primals_952, primals_953, primals_954, primals_955, primals_957, primals_958, primals_959, primals_960, primals_962, primals_963, primals_964, primals_965, primals_967, primals_968, primals_969, primals_970, primals_972, primals_973, primals_974, primals_975, primals_977, primals_978, primals_979, primals_980, primals_982, primals_983, primals_984, primals_985, primals_987, primals_988, primals_989, primals_990, primals_992, primals_993, primals_994, primals_995, primals_997, primals_998, primals_999, primals_1000, primals_1002, primals_1003, primals_1004, primals_1005, primals_1007, primals_1008, primals_1009, primals_1010, primals_1012, primals_1013, primals_1014, primals_1015, primals_1017, primals_1018, primals_1019, primals_1020, primals_1022, primals_1023, primals_1024, primals_1025, primals_1027, primals_1028, primals_1029, primals_1030, primals_1032, primals_1033, primals_1034, primals_1035, primals_1037, primals_1038, primals_1039, primals_1040, primals_1042, primals_1043, primals_1044, primals_1045, primals_1047, primals_1048, primals_1049, primals_1050, primals_1052, primals_1053, primals_1054, primals_1055, primals_1057, primals_1058, primals_1059, primals_1060, primals_1062, primals_1063, primals_1064, primals_1065, primals_1067, primals_1068, primals_1069, primals_1070, primals_1072, primals_1073, primals_1074, primals_1075, primals_1077, primals_1078, primals_1079, primals_1080, primals_1082, primals_1083, primals_1084, primals_1085, primals_1087, primals_1088, primals_1089, primals_1090, primals_1092, primals_1093, primals_1094, primals_1095, primals_1097, primals_1098, primals_1099, primals_1100, primals_1102, primals_1103, primals_1104, primals_1105, primals_1107, primals_1108, primals_1109, primals_1110, primals_1112, primals_1113, primals_1114, primals_1115, primals_1117, primals_1118, primals_1119, primals_1120, primals_1122, primals_1123, primals_1124, primals_1125, primals_1127, primals_1128, primals_1129, primals_1130, primals_1132, primals_1133, primals_1134, primals_1135, primals_1137, primals_1138, primals_1139, primals_1140, primals_1142, primals_1143, primals_1144, primals_1145, primals_1147, primals_1148, primals_1149, primals_1150, primals_1152, primals_1153, primals_1154, primals_1155, primals_1157, primals_1158, primals_1159, primals_1160, primals_1162, primals_1163, primals_1164, primals_1165, primals_1167, primals_1168, primals_1169, primals_1170, primals_1172, primals_1173, primals_1174, primals_1175, primals_1177, primals_1178, primals_1179, primals_1180, primals_1182, primals_1183, primals_1184, primals_1185, primals_1187, primals_1188, primals_1189, primals_1190, primals_1192, primals_1193, primals_1194, primals_1195, primals_1197, primals_1198, primals_1199, primals_1200, primals_1202, primals_1203, primals_1204, primals_1205, primals_1207, primals_1208, primals_1209, primals_1210, primals_1212, primals_1213, primals_1214, primals_1215, primals_1217, primals_1218, primals_1219, primals_1220, primals_1222, primals_1223, primals_1224, primals_1225, primals_1227, primals_1228, primals_1229, primals_1230, primals_1232, primals_1233, primals_1234, primals_1235, primals_1237, primals_1238, primals_1239, primals_1240, primals_1242, primals_1243, primals_1244, primals_1245, primals_1247, primals_1248, primals_1249, primals_1250, primals_1252, primals_1253, primals_1254, primals_1255, primals_1257, primals_1258, primals_1259, primals_1260, primals_1262, primals_1263, primals_1264, primals_1265, primals_1267, primals_1268, primals_1269, primals_1270, primals_1272, primals_1273, primals_1274, primals_1275, primals_1277, primals_1278, primals_1279, primals_1280, primals_1282, primals_1283, primals_1284, primals_1285, primals_1287, primals_1288, primals_1289, primals_1290, primals_1292, primals_1293, primals_1294, primals_1295, primals_1297, primals_1298, primals_1299, primals_1300, primals_1302, primals_1303, primals_1304, primals_1305, primals_1307, primals_1308, primals_1309, primals_1310, primals_1312, primals_1313, primals_1314, primals_1315, primals_1317, primals_1318, primals_1319, primals_1320, primals_1322, primals_1323, primals_1324, primals_1325, primals_1327, primals_1328, primals_1329, primals_1330, primals_1332, primals_1333, primals_1334, primals_1335, primals_1337, primals_1338, primals_1339, primals_1340, primals_1342, primals_1343, primals_1344, primals_1345, primals_1347, primals_1348, primals_1349, primals_1350, primals_1352, primals_1353, primals_1354, primals_1355, primals_1357, primals_1358, primals_1359, primals_1360, primals_1362, primals_1363, primals_1364, primals_1365, primals_1367, primals_1368, primals_1369, primals_1370, primals_1372, primals_1373, primals_1374, primals_1375, primals_1377, primals_1378, primals_1379, primals_1380, primals_1382, primals_1383, primals_1384, primals_1385, primals_1387, primals_1388, primals_1389, primals_1390, primals_1392, primals_1393, primals_1394, primals_1395, primals_1397, primals_1398, primals_1399, primals_1400, primals_1402, primals_1403, primals_1404, primals_1405, primals_1407, primals_1408, primals_1409, primals_1410, primals_1412, primals_1413, primals_1414, primals_1415, primals_1417, primals_1418, primals_1419, primals_1420, primals_1422, primals_1423, primals_1424, primals_1425, primals_1427, primals_1428, primals_1429, primals_1430, primals_1432, primals_1433, primals_1434, primals_1435, primals_1437, primals_1438, primals_1439, primals_1440, primals_1442, primals_1443, primals_1444, primals_1445, primals_1447, primals_1448, primals_1449, primals_1450, primals_1452, primals_1453, primals_1454, primals_1455, primals_1457, primals_1458, primals_1459, primals_1460, primals_1462, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf125, buf126, buf127, buf128, buf130, buf131, buf132, buf133, buf134, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf188, buf189, buf190, buf192, buf193, buf194, buf195, buf196, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf246, buf247, buf248, buf250, buf251, buf252, buf254, buf255, buf256, buf257, buf258, buf260, buf261, buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf273, buf274, buf275, buf276, buf277, buf278, buf279, buf280, buf281, buf282, buf283, buf284, buf285, buf286, buf287, buf288, buf289, buf290, buf291, buf292, buf293, buf294, buf295, buf296, buf297, buf298, buf299, buf300, buf301, buf302, buf303, buf304, buf305, buf306, buf307, buf308, buf309, buf310, buf312, buf313, buf314, buf316, buf317, buf318, buf319, buf320, buf322, buf323, buf324, buf325, buf326, buf327, buf328, buf329, buf330, buf331, buf332, buf333, buf334, buf335, buf336, buf337, buf338, buf339, buf340, buf341, buf342, buf343, buf344, buf345, buf346, buf347, buf348, buf349, buf350, buf351, buf352, buf353, buf354, buf355, buf356, buf357, buf358, buf359, buf360, buf361, buf362, buf363, buf364, buf365, buf366, buf367, buf368, buf369, buf370, buf371, buf372, buf373, buf374, buf375, buf376, buf377, buf378, buf379, buf380, buf381, buf382, buf383, buf384, buf385, buf386, buf387, buf388, buf389, buf390, buf391, buf392, buf394, buf395, buf396, buf397, buf398, buf400, buf401, buf402, buf403, buf404, buf405, buf406, buf409, buf410, buf411, buf412, buf413, buf414, buf415, buf416, buf417, buf418, buf420, buf421, buf422, buf423, buf424, buf425, buf426, buf427, buf428, buf429, buf430, buf431, buf432, buf433, buf434, buf435, buf436, buf437, buf438, buf439, buf440, buf441, buf442, buf443, buf444, buf445, buf446, buf447, buf448, buf449, buf450, buf451, buf452, buf453, buf454, buf455, buf456, buf457, buf458, buf459, buf460, buf461, buf462, buf463, buf464, buf465, buf466, buf467, buf468, buf469, buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf477, buf478, buf479, buf480, buf481, buf482, buf483, buf484, buf485, buf486, buf487, buf489, buf490, buf491, buf492, buf494, buf495, buf496, buf497, buf498, buf499, buf502, buf503, buf504, buf505, buf506, buf507, buf508, buf509, buf510, buf511, buf513, buf514, buf515, buf516, buf517, buf518, buf519, buf520, buf521, buf522, buf523, buf524, buf525, buf526, buf527, buf528, buf530, buf531, buf532, buf533, buf534, buf535, buf536, buf537, buf538, buf539, buf540, buf541, buf542, buf543, buf544, buf545, buf546, buf547, buf548, buf549, buf550, buf551, buf552, buf553, buf554, buf555, buf556, buf557, buf558, buf559, buf560, buf561, buf562, buf563, buf564, buf565, buf566, buf567, buf568, buf569, buf570, buf571, buf572, buf573, buf574, buf575, buf576, buf577, buf578, buf579, buf580, buf582, buf585, )


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
    primals_77 = rand_strided((32, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1106 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1107 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1112 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1117 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1118 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1122 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1124 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1126 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1127 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1132 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1137 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1142 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1147 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1152 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1157 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1162 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1167 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1168 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1172 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1177 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1181 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1182 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1186 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1187 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1192 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1197 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1202 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1205 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1206 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1207 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1212 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1217 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1218 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1222 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1225 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1226 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1227 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1231 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1232 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1234 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1237 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1238 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1239 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1240 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1241 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1242 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1247 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1249 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1252 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1257 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1259 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1260 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1262 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1267 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1269 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1270 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1272 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1276 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1277 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1278 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1279 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1282 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1287 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1290 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1291 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1292 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1293 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1296 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1297 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1299 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1302 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1306 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1307 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1308 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1312 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1313 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1314 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1315 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1316 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1317 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1318 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1320 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1321 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1322 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1323 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1327 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1328 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1332 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1337 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1338 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1342 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1344 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1347 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1350 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1352 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1354 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1355 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1357 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1358 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1362 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1367 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1368 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1369 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1370 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1372 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1373 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1374 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1377 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1382 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1385 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1386 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1387 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1389 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1390 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1391 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1392 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1397 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1398 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1402 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1403 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1404 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1406 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1407 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1408 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1409 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1410 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1411 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1412 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1413 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1414 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1415 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1416 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1417 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1418 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1419 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1420 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1421 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1422 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1423 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1424 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1425 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1426 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1427 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1428 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1429 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1430 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1431 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1432 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1433 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1434 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1435 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1436 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1437 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1438 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1439 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1441 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1442 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1445 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1446 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1447 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1448 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1449 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1450 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1451 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1452 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1453 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1454 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1455 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1456 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1457 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1458 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1459 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1460 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1461 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1462 = rand_strided((17, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1463 = rand_strided((17, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
