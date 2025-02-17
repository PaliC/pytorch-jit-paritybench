# AOT ID: ['28_forward']
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


# kernel path: inductor_cache/o6/co6geao5f2vm45uxh3g2u264evdjf4zjyzcd5etrmr2ep5xk4337.py
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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 508032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3969) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coeioavgkyr35mjbsh7cov5tn2oblspaqwjwsimgwq3pidmk522s.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_4, mul_5, sub_1
#   x_5 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3721) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ky/ckyg3cj3zfkvsy26hwoe6nifauqaifuosr3jr2nmll2zrvlgi674.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_7 => add_5, mul_7, mul_8, sub_2
#   x_8 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 952576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3721) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjmub2szprnbs5f5pvcf3l6njsoyyebqamqctnjggwi55hrhdlq.py
# Topologically Sorted Source Nodes: [x0], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x0 => _low_memory_max_pool2d_with_offsets, getitem_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 30)
    x1 = ((xindex // 30) % 30)
    x4 = xindex // 900
    x3 = xindex // 57600
    x5 = (xindex % 57600)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (61 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (62 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (63 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (122 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (123 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (124 + 2*x0 + 122*x1 + 3721*x4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x5 + 144000*x3), tmp16, xmask)
    tl.store(out_ptr1 + (x6), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/conqrlw4hrn6uoqpkzez5z6r5u2u5x5cqxmwbmxe3vd2dw7rjjlp.py
# Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_10 => add_7, mul_10, mul_11, sub_3
#   x_11 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 345600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 900) % 96)
    x2 = xindex // 86400
    x4 = (xindex % 86400)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 144000*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q2/cq2krzxcis6io6dub475rpxhp6koyzra76olkunblygcmb6yqus2.py
# Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_13 => add_9, mul_13, mul_14, sub_4
#   x_14 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 900) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/55/c55albente4jt5qsjrvf4lyichmsrsdqcvlkziilig3rjsyxl7zs.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_9], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 784) % 192)
    x0 = (xindex % 784)
    x2 = xindex // 150528
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 784*(x1) + 75264*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 192, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 784*((-96) + x1) + 75264*x2), tmp25, other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-96) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-96) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-96) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-96) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/gs/cgsdudxhyjcw4vsi3y3frr2fyqpm5bfvj7z5jnb5h4w4rlypvwuy.py
# Topologically Sorted Source Nodes: [x1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x1 => _low_memory_max_pool2d_with_offsets_1, getitem_3
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_1, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 13)
    x1 = ((xindex // 13) % 13)
    x4 = xindex // 169
    x3 = xindex // 32448
    x5 = (xindex % 32448)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (28 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (29 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (30 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (56 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (57 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (58 + 2*x0 + 56*x1 + 784*x4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x5 + 64896*x3), tmp16, xmask)
    tl.store(out_ptr1 + (x5 + 32512*x3), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bh/cbhmeavlcnigumavnp73u6vzd3h5vyzldtrnt6jnnpzk3uljnk7o.py
# Topologically Sorted Source Nodes: [x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_31 => add_21, mul_31, mul_32, sub_10
#   x_32 => relu_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 169) % 192)
    x2 = xindex // 32448
    x4 = (xindex % 32448)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 64896*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2oqy7evnlkjat6nqco5pbih4ea4dbaq45rhyncqwpolz6rdhc4.py
# Topologically Sorted Source Nodes: [x_37, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_37 => add_25, mul_37, mul_38, sub_12
#   x_38 => relu_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
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
    xnumel = 43264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 169) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mr/cmrm7sue7cxo7sclncvregrhzel73qkfcwrfxiasw3bgq4zqmih3.py
# Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_46 => add_31, mul_46, mul_47, sub_15
#   x_47 => relu_15
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 169) % 96)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p5/cp56qirjwzmde6rqqiphnsxbzy2q6tdkyoop7ta7vhjlkzxx4afi.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_2, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_11 = async_compile.triton('triton_poi_fused_avg_pool2d_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 259584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 13) % 13)
    x0 = (xindex % 13)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14) + x4), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-13) + x4), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-12) + x4), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (12 + x4), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (13 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14 + x4), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))) + ((13) * ((13) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (13)))*((13) * ((13) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (13))) + ((-1)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((13) * ((13) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (13)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((13) * ((13) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (13))))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e2/ce2m57mrmkbvyhyf3gqdszstmzh5k5wfbxnq77rdtf3s3bj2h5ps.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_11, %relu_13, %relu_16, %relu_17], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 259584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 169) % 384)
    x0 = (xindex % 169)
    x2 = xindex // 64896
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 169*(x1) + 16224*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
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
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 192, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 169*((-96) + x1) + 16224*x2), tmp28 & xmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-96) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-96) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-96) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-96) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 288, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (x0 + 169*((-192) + x1) + 16224*x2), tmp52 & xmask, other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-192) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-192) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-192) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-192) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 384, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (x0 + 169*((-288) + x1) + 16224*x2), tmp73 & xmask, other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-288) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-288) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-288) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-288) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x3), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i2/ci2qoytm4lkxqg2enyptrqtjx4lb6hiue4xatchm6pbycefvgx3v.py
# Topologically Sorted Source Nodes: [x_121, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_121 => add_81, mul_121, mul_122, sub_40
#   x_122 => relu_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_321), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %unsqueeze_325), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %unsqueeze_327), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_81,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 169) % 192)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfebwj7ukj2cobsemcc26tr24tfzumnghqgm2ev5dkmjh2fbdbdr.py
# Topologically Sorted Source Nodes: [x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_124 => add_83, mul_124, mul_125, sub_41
#   x_125 => relu_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 169) % 224)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2pnomwh7hc4r4jay4metyfza2rwsw62m7krp66lpdwp5p4d5x5.py
# Topologically Sorted Source Nodes: [x2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x2 => _low_memory_max_pool2d_with_offsets_2, getitem_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_6, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_15 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_15(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x4 = xindex // 36
    x3 = xindex // 13824
    x5 = (xindex % 13824)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (27 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + 2*x0 + 26*x1 + 169*x4), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x5 + 36864*x3), tmp16, xmask)
    tl.store(out_ptr1 + (x6), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5c/c5cpcwppz4caz4haxv7kcbgqqultnjm4v5jvzmvmbqaiae4hsjdo.py
# Topologically Sorted Source Nodes: [x_118, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_118 => add_79, mul_118, mul_119, sub_39
#   x_119 => relu_39
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 36) % 384)
    x2 = xindex // 13824
    x4 = (xindex % 13824)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 36864*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmq4kixcf2em4l5meeluuvv5ehz7tp4nlfqqzqrjdefpoiaidpky.py
# Topologically Sorted Source Nodes: [x_127, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_127 => add_85, mul_127, mul_128, sub_42
#   x_128 => relu_42
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_42 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 36) % 256)
    x2 = xindex // 9216
    x4 = (xindex % 9216)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 36864*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmgep3t4wrm42t5adgm36hd5cg537xs3nmcc56udiqlcoalhnb2u.py
# Topologically Sorted Source Nodes: [x_133, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_133 => add_89, mul_133, mul_134, sub_44
#   x_134 => relu_44
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_357), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_359), kwargs = {})
#   %relu_44 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_89,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 36) % 192)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nw/cnwenzympdknk5pwjzx22yybsjk4uoxyuzt3mcivr6ryf5pri5wp.py
# Topologically Sorted Source Nodes: [x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_136 => add_91, mul_136, mul_137, sub_45
#   x_137 => relu_45
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 36) % 224)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3k/c3kog3gu4ssrz2uimt7qyjpvta67aiu4ddnnd6h7wypw6sdilpf4.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_5 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_7, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_poi_fused_avg_pool2d_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7) + x4), tmp10, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-6) + x4), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5) + x4), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5 + x4), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (6 + x4), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7 + x4), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))) + ((6) * ((6) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (6)))*((6) * ((6) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (6))) + ((-1)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((6) * ((6) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (6)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((6) * ((6) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (6))))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cww4cuiudkpmwqxoc36iihypgfobbcg5cjp6a7opyaifhwcsa2jo.py
# Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_8 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_43, %relu_46, %relu_51, %relu_52], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 36) % 1024)
    x0 = (xindex % 36)
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 36*(x1) + 13824*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 640, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 36*((-384) + x1) + 9216*x2), tmp28, other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 896, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (x0 + 36*((-640) + x1) + 9216*x2), tmp52, other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 1024, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (x0 + 36*((-896) + x1) + 4608*x2), tmp73, other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x3), tmp98, None)
''', device_str='cuda')


# kernel path: inductor_cache/2w/c2wvra7yz3otxrxhkuy44h7ypgeprqko3rdjo7zzdmgdliyonjta.py
# Topologically Sorted Source Nodes: [x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_346 => add_231, mul_346, mul_347, sub_115
#   x_347 => relu_115
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %relu_115 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_231,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 36) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxlveoyzfqepnu53o2qcxtlg4vydrr3zbqzy6luqzogejx7e2uk.py
# Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_352 => add_235, mul_352, mul_353, sub_117
#   x_353 => relu_117
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %relu_117 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_235,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 36) % 320)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci34orjsnf45xepm5xxy3svymu6oexbbzz36dthh4vtjmg6ynzif.py
# Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x2_1 => _low_memory_max_pool2d_with_offsets_3, getitem_7
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_14, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_24 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_24(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x3 = xindex // 4096
    x5 = (xindex % 4096)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (6 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (7 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (8 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (13 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (14 + 2*x0 + 12*x1 + 36*x4), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x5 + 6144*x3), tmp16, None)
    tl.store(out_ptr1 + (x6), tmp41, None)
''', device_str='cuda')


# kernel path: inductor_cache/kn/ckn6sx3sbokx2jp2tsayrekjyhng2sz6euruzfpdk4gzy5on4rcn.py
# Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_343 => add_229, mul_343, mul_344, sub_114
#   x_344 => relu_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %relu_114 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 192)
    x2 = xindex // 768
    x4 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 6144*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnj4hsfusuv5xezgorsvagivnodljhamqehqikfqufh5n6hkyl4.py
# Topologically Sorted Source Nodes: [x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_355 => add_237, mul_355, mul_356, sub_118
#   x_356 => relu_118
# Graph fragment:
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_945), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_949), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_951), kwargs = {})
#   %relu_118 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 320)
    x2 = xindex // 1280
    x4 = (xindex % 1280)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 6144*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vs/cvshb6odfsy5gnrg7fkfnhlrwc7frd5n32ie2cznnpsygbqiip7l.py
# Topologically Sorted Source Nodes: [x_361, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_361 => add_241, mul_361, mul_362, sub_120
#   x_362 => relu_120
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_961), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_965), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_967), kwargs = {})
#   %relu_120 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_241,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 384)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eg/cegaem7txcskm536fvtm7qqla6xxigff2novbb2iwbscejxm42mr.py
# Topologically Sorted Source Nodes: [x1_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x1_1 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_121, %relu_122], 1), kwargs = {})
triton_poi_fused_cat_28 = async_compile.triton('triton_poi_fused_cat_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 512)
    x0 = (xindex % 4)
    x2 = xindex // 2048
    x3 = (xindex % 2048)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 1024*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 512, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-256) + x1) + 1024*x2), tmp25, other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-256) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-256) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-256) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-256) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3 + 6144*x2), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxrrpb5p55eu5jyd2yxzbmjt3xwdd7teegl3aaabro2f6ttj7v4.py
# Topologically Sorted Source Nodes: [x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_373 => add_249, mul_373, mul_374, sub_124
#   x_374 => relu_124
# Graph fragment:
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_993), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_124, %unsqueeze_995), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_997), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_999), kwargs = {})
#   %relu_124 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_249,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 448)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cukfepg2c5zjgysxoa5vs2muietymslh4wqatrfamlsgm7xw6tyt.py
# Topologically Sorted Source Nodes: [x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_376 => add_251, mul_376, mul_377, sub_125
#   x_377 => relu_125
# Graph fragment:
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_1001), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %unsqueeze_1003), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %unsqueeze_1005), kwargs = {})
#   %add_251 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_377, %unsqueeze_1007), kwargs = {})
#   %relu_125 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_251,), kwargs = {})
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
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ly/clyt6ljigsdiq4bommvvmkdn57edhm6hdkynyaduxcxdlc4de33s.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_12 => avg_pool2d_11
# Graph fragment:
#   %avg_pool2d_11 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_15, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_31 = async_compile.triton('triton_poi_fused_avg_pool2d_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
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
    tmp11 = tl.load(in_ptr0 + ((-3) + x4), tmp10, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2) + x4), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-1) + x4), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (1 + x4), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2 + x4), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (3 + x4), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 4 + ((-2)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) + ((-2)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/ys/cysoqndhdcbhjcte3vf55uuhve6ozezot7zvk54akdx4iifjomyc.py
# Topologically Sorted Source Nodes: [x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_358 => add_239, mul_358, mul_359, sub_119
#   x_359 => relu_119
# Graph fragment:
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_953), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_957), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_359, %unsqueeze_959), kwargs = {})
#   %relu_119 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_239,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
    x2 = xindex // 1024
    x4 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + 6144*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/3q/c3qsl6o72lboevfursjicz56af42l4ikey2jmhzsfn7svhpukhii.py
# Topologically Sorted Source Nodes: [v], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   v => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_24, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_33 = async_compile.triton('triton_poi_fused_mean_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 128, 128), (49152, 16384, 128, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (64, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_28, (96, ), (1, ))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (64, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, 64, 1, 7), (448, 7, 7, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, 64, 7, 1), (448, 7, 1, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_48, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (96, ), (1, ))
    assert_size_stride(primals_51, (96, ), (1, ))
    assert_size_stride(primals_52, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_58, (96, ), (1, ))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_60, (96, ), (1, ))
    assert_size_stride(primals_61, (96, ), (1, ))
    assert_size_stride(primals_62, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_68, (96, ), (1, ))
    assert_size_stride(primals_69, (96, ), (1, ))
    assert_size_stride(primals_70, (96, ), (1, ))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_72, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_78, (96, ), (1, ))
    assert_size_stride(primals_79, (96, ), (1, ))
    assert_size_stride(primals_80, (96, ), (1, ))
    assert_size_stride(primals_81, (96, ), (1, ))
    assert_size_stride(primals_82, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_83, (96, ), (1, ))
    assert_size_stride(primals_84, (96, ), (1, ))
    assert_size_stride(primals_85, (96, ), (1, ))
    assert_size_stride(primals_86, (96, ), (1, ))
    assert_size_stride(primals_87, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_88, (96, ), (1, ))
    assert_size_stride(primals_89, (96, ), (1, ))
    assert_size_stride(primals_90, (96, ), (1, ))
    assert_size_stride(primals_91, (96, ), (1, ))
    assert_size_stride(primals_92, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_93, (96, ), (1, ))
    assert_size_stride(primals_94, (96, ), (1, ))
    assert_size_stride(primals_95, (96, ), (1, ))
    assert_size_stride(primals_96, (96, ), (1, ))
    assert_size_stride(primals_97, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_103, (96, ), (1, ))
    assert_size_stride(primals_104, (96, ), (1, ))
    assert_size_stride(primals_105, (96, ), (1, ))
    assert_size_stride(primals_106, (96, ), (1, ))
    assert_size_stride(primals_107, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_108, (64, ), (1, ))
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_111, (64, ), (1, ))
    assert_size_stride(primals_112, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_113, (96, ), (1, ))
    assert_size_stride(primals_114, (96, ), (1, ))
    assert_size_stride(primals_115, (96, ), (1, ))
    assert_size_stride(primals_116, (96, ), (1, ))
    assert_size_stride(primals_117, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_118, (96, ), (1, ))
    assert_size_stride(primals_119, (96, ), (1, ))
    assert_size_stride(primals_120, (96, ), (1, ))
    assert_size_stride(primals_121, (96, ), (1, ))
    assert_size_stride(primals_122, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_123, (96, ), (1, ))
    assert_size_stride(primals_124, (96, ), (1, ))
    assert_size_stride(primals_125, (96, ), (1, ))
    assert_size_stride(primals_126, (96, ), (1, ))
    assert_size_stride(primals_127, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_128, (96, ), (1, ))
    assert_size_stride(primals_129, (96, ), (1, ))
    assert_size_stride(primals_130, (96, ), (1, ))
    assert_size_stride(primals_131, (96, ), (1, ))
    assert_size_stride(primals_132, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, ), (1, ))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_138, (96, ), (1, ))
    assert_size_stride(primals_139, (96, ), (1, ))
    assert_size_stride(primals_140, (96, ), (1, ))
    assert_size_stride(primals_141, (96, ), (1, ))
    assert_size_stride(primals_142, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, ), (1, ))
    assert_size_stride(primals_147, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_148, (96, ), (1, ))
    assert_size_stride(primals_149, (96, ), (1, ))
    assert_size_stride(primals_150, (96, ), (1, ))
    assert_size_stride(primals_151, (96, ), (1, ))
    assert_size_stride(primals_152, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_153, (96, ), (1, ))
    assert_size_stride(primals_154, (96, ), (1, ))
    assert_size_stride(primals_155, (96, ), (1, ))
    assert_size_stride(primals_156, (96, ), (1, ))
    assert_size_stride(primals_157, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_158, (96, ), (1, ))
    assert_size_stride(primals_159, (96, ), (1, ))
    assert_size_stride(primals_160, (96, ), (1, ))
    assert_size_stride(primals_161, (96, ), (1, ))
    assert_size_stride(primals_162, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_163, (96, ), (1, ))
    assert_size_stride(primals_164, (96, ), (1, ))
    assert_size_stride(primals_165, (96, ), (1, ))
    assert_size_stride(primals_166, (96, ), (1, ))
    assert_size_stride(primals_167, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, ), (1, ))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_173, (96, ), (1, ))
    assert_size_stride(primals_174, (96, ), (1, ))
    assert_size_stride(primals_175, (96, ), (1, ))
    assert_size_stride(primals_176, (96, ), (1, ))
    assert_size_stride(primals_177, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_183, (96, ), (1, ))
    assert_size_stride(primals_184, (96, ), (1, ))
    assert_size_stride(primals_185, (96, ), (1, ))
    assert_size_stride(primals_186, (96, ), (1, ))
    assert_size_stride(primals_187, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_188, (96, ), (1, ))
    assert_size_stride(primals_189, (96, ), (1, ))
    assert_size_stride(primals_190, (96, ), (1, ))
    assert_size_stride(primals_191, (96, ), (1, ))
    assert_size_stride(primals_192, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_193, (96, ), (1, ))
    assert_size_stride(primals_194, (96, ), (1, ))
    assert_size_stride(primals_195, (96, ), (1, ))
    assert_size_stride(primals_196, (96, ), (1, ))
    assert_size_stride(primals_197, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_198, (384, ), (1, ))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (224, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_208, (224, ), (1, ))
    assert_size_stride(primals_209, (224, ), (1, ))
    assert_size_stride(primals_210, (224, ), (1, ))
    assert_size_stride(primals_211, (224, ), (1, ))
    assert_size_stride(primals_212, (256, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_218, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, ), (1, ))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_223, (192, ), (1, ))
    assert_size_stride(primals_224, (192, ), (1, ))
    assert_size_stride(primals_225, (192, ), (1, ))
    assert_size_stride(primals_226, (192, ), (1, ))
    assert_size_stride(primals_227, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_228, (224, ), (1, ))
    assert_size_stride(primals_229, (224, ), (1, ))
    assert_size_stride(primals_230, (224, ), (1, ))
    assert_size_stride(primals_231, (224, ), (1, ))
    assert_size_stride(primals_232, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_238, (192, ), (1, ))
    assert_size_stride(primals_239, (192, ), (1, ))
    assert_size_stride(primals_240, (192, ), (1, ))
    assert_size_stride(primals_241, (192, ), (1, ))
    assert_size_stride(primals_242, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_243, (192, ), (1, ))
    assert_size_stride(primals_244, (192, ), (1, ))
    assert_size_stride(primals_245, (192, ), (1, ))
    assert_size_stride(primals_246, (192, ), (1, ))
    assert_size_stride(primals_247, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_248, (224, ), (1, ))
    assert_size_stride(primals_249, (224, ), (1, ))
    assert_size_stride(primals_250, (224, ), (1, ))
    assert_size_stride(primals_251, (224, ), (1, ))
    assert_size_stride(primals_252, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_253, (224, ), (1, ))
    assert_size_stride(primals_254, (224, ), (1, ))
    assert_size_stride(primals_255, (224, ), (1, ))
    assert_size_stride(primals_256, (224, ), (1, ))
    assert_size_stride(primals_257, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_269, (384, ), (1, ))
    assert_size_stride(primals_270, (384, ), (1, ))
    assert_size_stride(primals_271, (384, ), (1, ))
    assert_size_stride(primals_272, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (192, ), (1, ))
    assert_size_stride(primals_276, (192, ), (1, ))
    assert_size_stride(primals_277, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_278, (224, ), (1, ))
    assert_size_stride(primals_279, (224, ), (1, ))
    assert_size_stride(primals_280, (224, ), (1, ))
    assert_size_stride(primals_281, (224, ), (1, ))
    assert_size_stride(primals_282, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_288, (192, ), (1, ))
    assert_size_stride(primals_289, (192, ), (1, ))
    assert_size_stride(primals_290, (192, ), (1, ))
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_293, (192, ), (1, ))
    assert_size_stride(primals_294, (192, ), (1, ))
    assert_size_stride(primals_295, (192, ), (1, ))
    assert_size_stride(primals_296, (192, ), (1, ))
    assert_size_stride(primals_297, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_298, (224, ), (1, ))
    assert_size_stride(primals_299, (224, ), (1, ))
    assert_size_stride(primals_300, (224, ), (1, ))
    assert_size_stride(primals_301, (224, ), (1, ))
    assert_size_stride(primals_302, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_303, (224, ), (1, ))
    assert_size_stride(primals_304, (224, ), (1, ))
    assert_size_stride(primals_305, (224, ), (1, ))
    assert_size_stride(primals_306, (224, ), (1, ))
    assert_size_stride(primals_307, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (256, ), (1, ))
    assert_size_stride(primals_310, (256, ), (1, ))
    assert_size_stride(primals_311, (256, ), (1, ))
    assert_size_stride(primals_312, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_318, (384, ), (1, ))
    assert_size_stride(primals_319, (384, ), (1, ))
    assert_size_stride(primals_320, (384, ), (1, ))
    assert_size_stride(primals_321, (384, ), (1, ))
    assert_size_stride(primals_322, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_323, (192, ), (1, ))
    assert_size_stride(primals_324, (192, ), (1, ))
    assert_size_stride(primals_325, (192, ), (1, ))
    assert_size_stride(primals_326, (192, ), (1, ))
    assert_size_stride(primals_327, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_328, (224, ), (1, ))
    assert_size_stride(primals_329, (224, ), (1, ))
    assert_size_stride(primals_330, (224, ), (1, ))
    assert_size_stride(primals_331, (224, ), (1, ))
    assert_size_stride(primals_332, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_333, (256, ), (1, ))
    assert_size_stride(primals_334, (256, ), (1, ))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_336, (256, ), (1, ))
    assert_size_stride(primals_337, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (192, ), (1, ))
    assert_size_stride(primals_340, (192, ), (1, ))
    assert_size_stride(primals_341, (192, ), (1, ))
    assert_size_stride(primals_342, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_343, (192, ), (1, ))
    assert_size_stride(primals_344, (192, ), (1, ))
    assert_size_stride(primals_345, (192, ), (1, ))
    assert_size_stride(primals_346, (192, ), (1, ))
    assert_size_stride(primals_347, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_348, (224, ), (1, ))
    assert_size_stride(primals_349, (224, ), (1, ))
    assert_size_stride(primals_350, (224, ), (1, ))
    assert_size_stride(primals_351, (224, ), (1, ))
    assert_size_stride(primals_352, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_353, (224, ), (1, ))
    assert_size_stride(primals_354, (224, ), (1, ))
    assert_size_stride(primals_355, (224, ), (1, ))
    assert_size_stride(primals_356, (224, ), (1, ))
    assert_size_stride(primals_357, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_358, (256, ), (1, ))
    assert_size_stride(primals_359, (256, ), (1, ))
    assert_size_stride(primals_360, (256, ), (1, ))
    assert_size_stride(primals_361, (256, ), (1, ))
    assert_size_stride(primals_362, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_368, (384, ), (1, ))
    assert_size_stride(primals_369, (384, ), (1, ))
    assert_size_stride(primals_370, (384, ), (1, ))
    assert_size_stride(primals_371, (384, ), (1, ))
    assert_size_stride(primals_372, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_373, (192, ), (1, ))
    assert_size_stride(primals_374, (192, ), (1, ))
    assert_size_stride(primals_375, (192, ), (1, ))
    assert_size_stride(primals_376, (192, ), (1, ))
    assert_size_stride(primals_377, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_378, (224, ), (1, ))
    assert_size_stride(primals_379, (224, ), (1, ))
    assert_size_stride(primals_380, (224, ), (1, ))
    assert_size_stride(primals_381, (224, ), (1, ))
    assert_size_stride(primals_382, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_383, (256, ), (1, ))
    assert_size_stride(primals_384, (256, ), (1, ))
    assert_size_stride(primals_385, (256, ), (1, ))
    assert_size_stride(primals_386, (256, ), (1, ))
    assert_size_stride(primals_387, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_388, (192, ), (1, ))
    assert_size_stride(primals_389, (192, ), (1, ))
    assert_size_stride(primals_390, (192, ), (1, ))
    assert_size_stride(primals_391, (192, ), (1, ))
    assert_size_stride(primals_392, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_393, (192, ), (1, ))
    assert_size_stride(primals_394, (192, ), (1, ))
    assert_size_stride(primals_395, (192, ), (1, ))
    assert_size_stride(primals_396, (192, ), (1, ))
    assert_size_stride(primals_397, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_398, (224, ), (1, ))
    assert_size_stride(primals_399, (224, ), (1, ))
    assert_size_stride(primals_400, (224, ), (1, ))
    assert_size_stride(primals_401, (224, ), (1, ))
    assert_size_stride(primals_402, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_403, (224, ), (1, ))
    assert_size_stride(primals_404, (224, ), (1, ))
    assert_size_stride(primals_405, (224, ), (1, ))
    assert_size_stride(primals_406, (224, ), (1, ))
    assert_size_stride(primals_407, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_408, (256, ), (1, ))
    assert_size_stride(primals_409, (256, ), (1, ))
    assert_size_stride(primals_410, (256, ), (1, ))
    assert_size_stride(primals_411, (256, ), (1, ))
    assert_size_stride(primals_412, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_413, (128, ), (1, ))
    assert_size_stride(primals_414, (128, ), (1, ))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_418, (384, ), (1, ))
    assert_size_stride(primals_419, (384, ), (1, ))
    assert_size_stride(primals_420, (384, ), (1, ))
    assert_size_stride(primals_421, (384, ), (1, ))
    assert_size_stride(primals_422, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_423, (192, ), (1, ))
    assert_size_stride(primals_424, (192, ), (1, ))
    assert_size_stride(primals_425, (192, ), (1, ))
    assert_size_stride(primals_426, (192, ), (1, ))
    assert_size_stride(primals_427, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_428, (224, ), (1, ))
    assert_size_stride(primals_429, (224, ), (1, ))
    assert_size_stride(primals_430, (224, ), (1, ))
    assert_size_stride(primals_431, (224, ), (1, ))
    assert_size_stride(primals_432, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_433, (256, ), (1, ))
    assert_size_stride(primals_434, (256, ), (1, ))
    assert_size_stride(primals_435, (256, ), (1, ))
    assert_size_stride(primals_436, (256, ), (1, ))
    assert_size_stride(primals_437, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_438, (192, ), (1, ))
    assert_size_stride(primals_439, (192, ), (1, ))
    assert_size_stride(primals_440, (192, ), (1, ))
    assert_size_stride(primals_441, (192, ), (1, ))
    assert_size_stride(primals_442, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_443, (192, ), (1, ))
    assert_size_stride(primals_444, (192, ), (1, ))
    assert_size_stride(primals_445, (192, ), (1, ))
    assert_size_stride(primals_446, (192, ), (1, ))
    assert_size_stride(primals_447, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_448, (224, ), (1, ))
    assert_size_stride(primals_449, (224, ), (1, ))
    assert_size_stride(primals_450, (224, ), (1, ))
    assert_size_stride(primals_451, (224, ), (1, ))
    assert_size_stride(primals_452, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_453, (224, ), (1, ))
    assert_size_stride(primals_454, (224, ), (1, ))
    assert_size_stride(primals_455, (224, ), (1, ))
    assert_size_stride(primals_456, (224, ), (1, ))
    assert_size_stride(primals_457, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_458, (256, ), (1, ))
    assert_size_stride(primals_459, (256, ), (1, ))
    assert_size_stride(primals_460, (256, ), (1, ))
    assert_size_stride(primals_461, (256, ), (1, ))
    assert_size_stride(primals_462, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (128, ), (1, ))
    assert_size_stride(primals_465, (128, ), (1, ))
    assert_size_stride(primals_466, (128, ), (1, ))
    assert_size_stride(primals_467, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_468, (384, ), (1, ))
    assert_size_stride(primals_469, (384, ), (1, ))
    assert_size_stride(primals_470, (384, ), (1, ))
    assert_size_stride(primals_471, (384, ), (1, ))
    assert_size_stride(primals_472, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_473, (192, ), (1, ))
    assert_size_stride(primals_474, (192, ), (1, ))
    assert_size_stride(primals_475, (192, ), (1, ))
    assert_size_stride(primals_476, (192, ), (1, ))
    assert_size_stride(primals_477, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_478, (224, ), (1, ))
    assert_size_stride(primals_479, (224, ), (1, ))
    assert_size_stride(primals_480, (224, ), (1, ))
    assert_size_stride(primals_481, (224, ), (1, ))
    assert_size_stride(primals_482, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_483, (256, ), (1, ))
    assert_size_stride(primals_484, (256, ), (1, ))
    assert_size_stride(primals_485, (256, ), (1, ))
    assert_size_stride(primals_486, (256, ), (1, ))
    assert_size_stride(primals_487, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_488, (192, ), (1, ))
    assert_size_stride(primals_489, (192, ), (1, ))
    assert_size_stride(primals_490, (192, ), (1, ))
    assert_size_stride(primals_491, (192, ), (1, ))
    assert_size_stride(primals_492, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_493, (192, ), (1, ))
    assert_size_stride(primals_494, (192, ), (1, ))
    assert_size_stride(primals_495, (192, ), (1, ))
    assert_size_stride(primals_496, (192, ), (1, ))
    assert_size_stride(primals_497, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_498, (224, ), (1, ))
    assert_size_stride(primals_499, (224, ), (1, ))
    assert_size_stride(primals_500, (224, ), (1, ))
    assert_size_stride(primals_501, (224, ), (1, ))
    assert_size_stride(primals_502, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_503, (224, ), (1, ))
    assert_size_stride(primals_504, (224, ), (1, ))
    assert_size_stride(primals_505, (224, ), (1, ))
    assert_size_stride(primals_506, (224, ), (1, ))
    assert_size_stride(primals_507, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_508, (256, ), (1, ))
    assert_size_stride(primals_509, (256, ), (1, ))
    assert_size_stride(primals_510, (256, ), (1, ))
    assert_size_stride(primals_511, (256, ), (1, ))
    assert_size_stride(primals_512, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_513, (128, ), (1, ))
    assert_size_stride(primals_514, (128, ), (1, ))
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, ), (1, ))
    assert_size_stride(primals_517, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_518, (384, ), (1, ))
    assert_size_stride(primals_519, (384, ), (1, ))
    assert_size_stride(primals_520, (384, ), (1, ))
    assert_size_stride(primals_521, (384, ), (1, ))
    assert_size_stride(primals_522, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_523, (192, ), (1, ))
    assert_size_stride(primals_524, (192, ), (1, ))
    assert_size_stride(primals_525, (192, ), (1, ))
    assert_size_stride(primals_526, (192, ), (1, ))
    assert_size_stride(primals_527, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_528, (224, ), (1, ))
    assert_size_stride(primals_529, (224, ), (1, ))
    assert_size_stride(primals_530, (224, ), (1, ))
    assert_size_stride(primals_531, (224, ), (1, ))
    assert_size_stride(primals_532, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_533, (256, ), (1, ))
    assert_size_stride(primals_534, (256, ), (1, ))
    assert_size_stride(primals_535, (256, ), (1, ))
    assert_size_stride(primals_536, (256, ), (1, ))
    assert_size_stride(primals_537, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_538, (192, ), (1, ))
    assert_size_stride(primals_539, (192, ), (1, ))
    assert_size_stride(primals_540, (192, ), (1, ))
    assert_size_stride(primals_541, (192, ), (1, ))
    assert_size_stride(primals_542, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_543, (192, ), (1, ))
    assert_size_stride(primals_544, (192, ), (1, ))
    assert_size_stride(primals_545, (192, ), (1, ))
    assert_size_stride(primals_546, (192, ), (1, ))
    assert_size_stride(primals_547, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_548, (224, ), (1, ))
    assert_size_stride(primals_549, (224, ), (1, ))
    assert_size_stride(primals_550, (224, ), (1, ))
    assert_size_stride(primals_551, (224, ), (1, ))
    assert_size_stride(primals_552, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_553, (224, ), (1, ))
    assert_size_stride(primals_554, (224, ), (1, ))
    assert_size_stride(primals_555, (224, ), (1, ))
    assert_size_stride(primals_556, (224, ), (1, ))
    assert_size_stride(primals_557, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_558, (256, ), (1, ))
    assert_size_stride(primals_559, (256, ), (1, ))
    assert_size_stride(primals_560, (256, ), (1, ))
    assert_size_stride(primals_561, (256, ), (1, ))
    assert_size_stride(primals_562, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_563, (128, ), (1, ))
    assert_size_stride(primals_564, (128, ), (1, ))
    assert_size_stride(primals_565, (128, ), (1, ))
    assert_size_stride(primals_566, (128, ), (1, ))
    assert_size_stride(primals_567, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_568, (192, ), (1, ))
    assert_size_stride(primals_569, (192, ), (1, ))
    assert_size_stride(primals_570, (192, ), (1, ))
    assert_size_stride(primals_571, (192, ), (1, ))
    assert_size_stride(primals_572, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_573, (192, ), (1, ))
    assert_size_stride(primals_574, (192, ), (1, ))
    assert_size_stride(primals_575, (192, ), (1, ))
    assert_size_stride(primals_576, (192, ), (1, ))
    assert_size_stride(primals_577, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_578, (256, ), (1, ))
    assert_size_stride(primals_579, (256, ), (1, ))
    assert_size_stride(primals_580, (256, ), (1, ))
    assert_size_stride(primals_581, (256, ), (1, ))
    assert_size_stride(primals_582, (256, 256, 1, 7), (1792, 7, 7, 1))
    assert_size_stride(primals_583, (256, ), (1, ))
    assert_size_stride(primals_584, (256, ), (1, ))
    assert_size_stride(primals_585, (256, ), (1, ))
    assert_size_stride(primals_586, (256, ), (1, ))
    assert_size_stride(primals_587, (320, 256, 7, 1), (1792, 7, 1, 1))
    assert_size_stride(primals_588, (320, ), (1, ))
    assert_size_stride(primals_589, (320, ), (1, ))
    assert_size_stride(primals_590, (320, ), (1, ))
    assert_size_stride(primals_591, (320, ), (1, ))
    assert_size_stride(primals_592, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_593, (320, ), (1, ))
    assert_size_stride(primals_594, (320, ), (1, ))
    assert_size_stride(primals_595, (320, ), (1, ))
    assert_size_stride(primals_596, (320, ), (1, ))
    assert_size_stride(primals_597, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_598, (256, ), (1, ))
    assert_size_stride(primals_599, (256, ), (1, ))
    assert_size_stride(primals_600, (256, ), (1, ))
    assert_size_stride(primals_601, (256, ), (1, ))
    assert_size_stride(primals_602, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_603, (384, ), (1, ))
    assert_size_stride(primals_604, (384, ), (1, ))
    assert_size_stride(primals_605, (384, ), (1, ))
    assert_size_stride(primals_606, (384, ), (1, ))
    assert_size_stride(primals_607, (256, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_608, (256, ), (1, ))
    assert_size_stride(primals_609, (256, ), (1, ))
    assert_size_stride(primals_610, (256, ), (1, ))
    assert_size_stride(primals_611, (256, ), (1, ))
    assert_size_stride(primals_612, (256, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_613, (256, ), (1, ))
    assert_size_stride(primals_614, (256, ), (1, ))
    assert_size_stride(primals_615, (256, ), (1, ))
    assert_size_stride(primals_616, (256, ), (1, ))
    assert_size_stride(primals_617, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_618, (384, ), (1, ))
    assert_size_stride(primals_619, (384, ), (1, ))
    assert_size_stride(primals_620, (384, ), (1, ))
    assert_size_stride(primals_621, (384, ), (1, ))
    assert_size_stride(primals_622, (448, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_623, (448, ), (1, ))
    assert_size_stride(primals_624, (448, ), (1, ))
    assert_size_stride(primals_625, (448, ), (1, ))
    assert_size_stride(primals_626, (448, ), (1, ))
    assert_size_stride(primals_627, (512, 448, 1, 3), (1344, 3, 3, 1))
    assert_size_stride(primals_628, (512, ), (1, ))
    assert_size_stride(primals_629, (512, ), (1, ))
    assert_size_stride(primals_630, (512, ), (1, ))
    assert_size_stride(primals_631, (512, ), (1, ))
    assert_size_stride(primals_632, (256, 512, 1, 3), (1536, 3, 3, 1))
    assert_size_stride(primals_633, (256, ), (1, ))
    assert_size_stride(primals_634, (256, ), (1, ))
    assert_size_stride(primals_635, (256, ), (1, ))
    assert_size_stride(primals_636, (256, ), (1, ))
    assert_size_stride(primals_637, (256, 512, 3, 1), (1536, 3, 1, 1))
    assert_size_stride(primals_638, (256, ), (1, ))
    assert_size_stride(primals_639, (256, ), (1, ))
    assert_size_stride(primals_640, (256, ), (1, ))
    assert_size_stride(primals_641, (256, ), (1, ))
    assert_size_stride(primals_642, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_643, (256, ), (1, ))
    assert_size_stride(primals_644, (256, ), (1, ))
    assert_size_stride(primals_645, (256, ), (1, ))
    assert_size_stride(primals_646, (256, ), (1, ))
    assert_size_stride(primals_647, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_648, (256, ), (1, ))
    assert_size_stride(primals_649, (256, ), (1, ))
    assert_size_stride(primals_650, (256, ), (1, ))
    assert_size_stride(primals_651, (256, ), (1, ))
    assert_size_stride(primals_652, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_653, (384, ), (1, ))
    assert_size_stride(primals_654, (384, ), (1, ))
    assert_size_stride(primals_655, (384, ), (1, ))
    assert_size_stride(primals_656, (384, ), (1, ))
    assert_size_stride(primals_657, (256, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_658, (256, ), (1, ))
    assert_size_stride(primals_659, (256, ), (1, ))
    assert_size_stride(primals_660, (256, ), (1, ))
    assert_size_stride(primals_661, (256, ), (1, ))
    assert_size_stride(primals_662, (256, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_663, (256, ), (1, ))
    assert_size_stride(primals_664, (256, ), (1, ))
    assert_size_stride(primals_665, (256, ), (1, ))
    assert_size_stride(primals_666, (256, ), (1, ))
    assert_size_stride(primals_667, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_668, (384, ), (1, ))
    assert_size_stride(primals_669, (384, ), (1, ))
    assert_size_stride(primals_670, (384, ), (1, ))
    assert_size_stride(primals_671, (384, ), (1, ))
    assert_size_stride(primals_672, (448, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_673, (448, ), (1, ))
    assert_size_stride(primals_674, (448, ), (1, ))
    assert_size_stride(primals_675, (448, ), (1, ))
    assert_size_stride(primals_676, (448, ), (1, ))
    assert_size_stride(primals_677, (512, 448, 1, 3), (1344, 3, 3, 1))
    assert_size_stride(primals_678, (512, ), (1, ))
    assert_size_stride(primals_679, (512, ), (1, ))
    assert_size_stride(primals_680, (512, ), (1, ))
    assert_size_stride(primals_681, (512, ), (1, ))
    assert_size_stride(primals_682, (256, 512, 1, 3), (1536, 3, 3, 1))
    assert_size_stride(primals_683, (256, ), (1, ))
    assert_size_stride(primals_684, (256, ), (1, ))
    assert_size_stride(primals_685, (256, ), (1, ))
    assert_size_stride(primals_686, (256, ), (1, ))
    assert_size_stride(primals_687, (256, 512, 3, 1), (1536, 3, 1, 1))
    assert_size_stride(primals_688, (256, ), (1, ))
    assert_size_stride(primals_689, (256, ), (1, ))
    assert_size_stride(primals_690, (256, ), (1, ))
    assert_size_stride(primals_691, (256, ), (1, ))
    assert_size_stride(primals_692, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_693, (256, ), (1, ))
    assert_size_stride(primals_694, (256, ), (1, ))
    assert_size_stride(primals_695, (256, ), (1, ))
    assert_size_stride(primals_696, (256, ), (1, ))
    assert_size_stride(primals_697, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_698, (256, ), (1, ))
    assert_size_stride(primals_699, (256, ), (1, ))
    assert_size_stride(primals_700, (256, ), (1, ))
    assert_size_stride(primals_701, (256, ), (1, ))
    assert_size_stride(primals_702, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_703, (384, ), (1, ))
    assert_size_stride(primals_704, (384, ), (1, ))
    assert_size_stride(primals_705, (384, ), (1, ))
    assert_size_stride(primals_706, (384, ), (1, ))
    assert_size_stride(primals_707, (256, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(primals_708, (256, ), (1, ))
    assert_size_stride(primals_709, (256, ), (1, ))
    assert_size_stride(primals_710, (256, ), (1, ))
    assert_size_stride(primals_711, (256, ), (1, ))
    assert_size_stride(primals_712, (256, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_713, (256, ), (1, ))
    assert_size_stride(primals_714, (256, ), (1, ))
    assert_size_stride(primals_715, (256, ), (1, ))
    assert_size_stride(primals_716, (256, ), (1, ))
    assert_size_stride(primals_717, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_718, (384, ), (1, ))
    assert_size_stride(primals_719, (384, ), (1, ))
    assert_size_stride(primals_720, (384, ), (1, ))
    assert_size_stride(primals_721, (384, ), (1, ))
    assert_size_stride(primals_722, (448, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(primals_723, (448, ), (1, ))
    assert_size_stride(primals_724, (448, ), (1, ))
    assert_size_stride(primals_725, (448, ), (1, ))
    assert_size_stride(primals_726, (448, ), (1, ))
    assert_size_stride(primals_727, (512, 448, 1, 3), (1344, 3, 3, 1))
    assert_size_stride(primals_728, (512, ), (1, ))
    assert_size_stride(primals_729, (512, ), (1, ))
    assert_size_stride(primals_730, (512, ), (1, ))
    assert_size_stride(primals_731, (512, ), (1, ))
    assert_size_stride(primals_732, (256, 512, 1, 3), (1536, 3, 3, 1))
    assert_size_stride(primals_733, (256, ), (1, ))
    assert_size_stride(primals_734, (256, ), (1, ))
    assert_size_stride(primals_735, (256, ), (1, ))
    assert_size_stride(primals_736, (256, ), (1, ))
    assert_size_stride(primals_737, (256, 512, 3, 1), (1536, 3, 1, 1))
    assert_size_stride(primals_738, (256, ), (1, ))
    assert_size_stride(primals_739, (256, ), (1, ))
    assert_size_stride(primals_740, (256, ), (1, ))
    assert_size_stride(primals_741, (256, ), (1, ))
    assert_size_stride(primals_742, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_743, (256, ), (1, ))
    assert_size_stride(primals_744, (256, ), (1, ))
    assert_size_stride(primals_745, (256, ), (1, ))
    assert_size_stride(primals_746, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 63, 63), (127008, 3969, 63, 1))
        buf1 = empty_strided_cuda((4, 32, 63, 63), (127008, 3969, 63, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 508032, grid=grid(508032), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 61, 61), (119072, 3721, 61, 1))
        buf3 = empty_strided_cuda((4, 32, 61, 61), (119072, 3721, 61, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf2, primals_8, primals_9, primals_10, primals_11, buf3, 476288, grid=grid(476288), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 61, 61), (238144, 3721, 61, 1))
        buf5 = empty_strided_cuda((4, 64, 61, 61), (238144, 3721, 61, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_13, primals_14, primals_15, primals_16, buf5, 952576, grid=grid(952576), stream=stream0)
        del primals_16
        buf10 = empty_strided_cuda((4, 160, 30, 30), (144000, 900, 30, 1), torch.float32)
        buf6 = reinterpret_tensor(buf10, (4, 64, 30, 30), (144000, 900, 30, 1), 0)  # alias
        buf7 = empty_strided_cuda((4, 64, 30, 30), (57600, 900, 30, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x0], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf5, buf6, buf7, 230400, grid=grid(230400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf5, primals_17, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 96, 30, 30), (86400, 900, 30, 1))
        buf9 = reinterpret_tensor(buf10, (4, 96, 30, 30), (144000, 900, 30, 1), 57600)  # alias
        # Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, primals_18, primals_19, primals_20, primals_21, buf9, 345600, grid=grid(345600), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 30, 30), (57600, 900, 30, 1))
        buf12 = empty_strided_cuda((4, 64, 30, 30), (57600, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf11, primals_23, primals_24, primals_25, primals_26, buf12, 230400, grid=grid(230400), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 96, 28, 28), (75264, 784, 28, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf10, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 30, 30), (57600, 900, 30, 1))
        buf15 = empty_strided_cuda((4, 64, 30, 30), (57600, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf14, primals_33, primals_34, primals_35, primals_36, buf15, 230400, grid=grid(230400), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_37, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 30, 30), (57600, 900, 30, 1))
        buf17 = empty_strided_cuda((4, 64, 30, 30), (57600, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf16, primals_38, primals_39, primals_40, primals_41, buf17, 230400, grid=grid(230400), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_42, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 30, 30), (57600, 900, 30, 1))
        buf19 = empty_strided_cuda((4, 64, 30, 30), (57600, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf18, primals_43, primals_44, primals_45, primals_46, buf19, 230400, grid=grid(230400), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 96, 28, 28), (75264, 784, 28, 1))
        buf21 = empty_strided_cuda((4, 192, 28, 28), (150528, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf13, primals_28, primals_29, primals_30, primals_31, buf20, primals_48, primals_49, primals_50, primals_51, buf21, 602112, grid=grid(602112), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_52, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 192, 13, 13), (32448, 169, 13, 1))
        buf26 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        buf23 = reinterpret_tensor(buf26, (4, 192, 13, 13), (64896, 169, 13, 1), 32448)  # alias
        buf24 = empty_strided_cuda((4, 192, 13, 13), (32512, 169, 13, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf21, buf23, buf24, 129792, grid=grid(129792), stream=stream0)
        buf25 = reinterpret_tensor(buf26, (4, 192, 13, 13), (64896, 169, 13, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf22, primals_53, primals_54, primals_55, primals_56, buf25, 129792, grid=grid(129792), stream=stream0)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf29 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_37, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf28, primals_63, primals_64, primals_65, primals_66, buf29, 43264, grid=grid(43264), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf26, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf32 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf31, primals_73, primals_74, primals_75, primals_76, buf32, 43264, grid=grid(43264), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf34 = empty_strided_cuda((4, 96, 13, 13), (16224, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf33, primals_78, primals_79, primals_80, primals_81, buf34, 64896, grid=grid(64896), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf36 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_11.run(buf26, buf36, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf38 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf27, primals_58, primals_59, primals_60, primals_61, buf30, primals_68, primals_69, primals_70, primals_71, buf35, primals_83, primals_84, primals_85, primals_86, buf37, primals_88, primals_89, primals_90, primals_91, buf38, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf38, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf41 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_58, x_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf40, primals_98, primals_99, primals_100, primals_101, buf41, 43264, grid=grid(43264), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf38, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf44 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_64, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf43, primals_108, primals_109, primals_110, primals_111, buf44, 43264, grid=grid(43264), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf46 = empty_strided_cuda((4, 96, 13, 13), (16224, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf45, primals_113, primals_114, primals_115, primals_116, buf46, 64896, grid=grid(64896), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf48 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_11.run(buf38, buf48, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf50 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf39, primals_93, primals_94, primals_95, primals_96, buf42, primals_103, primals_104, primals_105, primals_106, buf47, primals_118, primals_119, primals_120, primals_121, buf49, primals_123, primals_124, primals_125, primals_126, buf50, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf50, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf53 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf52, primals_133, primals_134, primals_135, primals_136, buf53, 43264, grid=grid(43264), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf50, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf56 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_85, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf55, primals_143, primals_144, primals_145, primals_146, buf56, 43264, grid=grid(43264), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf58 = empty_strided_cuda((4, 96, 13, 13), (16224, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf57, primals_148, primals_149, primals_150, primals_151, buf58, 64896, grid=grid(64896), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf60 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_11.run(buf50, buf60, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf62 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf51, primals_128, primals_129, primals_130, primals_131, buf54, primals_138, primals_139, primals_140, primals_141, buf59, primals_153, primals_154, primals_155, primals_156, buf61, primals_158, primals_159, primals_160, primals_161, buf62, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf62, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf65 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_100, x_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf64, primals_168, primals_169, primals_170, primals_171, buf65, 43264, grid=grid(43264), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 96, 13, 13), (16224, 169, 13, 1))
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf62, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf68 = empty_strided_cuda((4, 64, 13, 13), (10816, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_106, x_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf67, primals_178, primals_179, primals_180, primals_181, buf68, 43264, grid=grid(43264), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf70 = empty_strided_cuda((4, 96, 13, 13), (16224, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_109, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf69, primals_183, primals_184, primals_185, primals_186, buf70, 64896, grid=grid(64896), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf72 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_11.run(buf62, buf72, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 96, 13, 13), (16224, 169, 13, 1))
        buf74 = empty_strided_cuda((4, 384, 13, 13), (64896, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf63, primals_163, primals_164, primals_165, primals_166, buf66, primals_173, primals_174, primals_175, primals_176, buf71, primals_188, primals_189, primals_190, primals_191, buf73, primals_193, primals_194, primals_195, primals_196, buf74, 259584, grid=grid(259584), stream=stream0)
        # Topologically Sorted Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_197, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf74, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 192, 13, 13), (32448, 169, 13, 1))
        buf77 = empty_strided_cuda((4, 192, 13, 13), (32448, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_121, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf76, primals_203, primals_204, primals_205, primals_206, buf77, 129792, grid=grid(129792), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 224, 13, 13), (37856, 169, 13, 1))
        buf79 = empty_strided_cuda((4, 224, 13, 13), (37856, 169, 13, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf78, primals_208, primals_209, primals_210, primals_211, buf79, 151424, grid=grid(151424), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_212, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf85 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        buf81 = reinterpret_tensor(buf85, (4, 384, 6, 6), (36864, 36, 6, 1), 23040)  # alias
        buf82 = empty_strided_cuda((4, 384, 6, 6), (13824, 36, 6, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_15.run(buf74, buf81, buf82, 55296, grid=grid(55296), stream=stream0)
        buf83 = reinterpret_tensor(buf85, (4, 384, 6, 6), (36864, 36, 6, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_118, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf75, primals_198, primals_199, primals_200, primals_201, buf83, 55296, grid=grid(55296), stream=stream0)
        buf84 = reinterpret_tensor(buf85, (4, 256, 6, 6), (36864, 36, 6, 1), 13824)  # alias
        # Topologically Sorted Source Nodes: [x_127, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf80, primals_213, primals_214, primals_215, primals_216, buf84, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf85, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf88 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_133, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf87, primals_223, primals_224, primals_225, primals_226, buf88, 27648, grid=grid(27648), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_227, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf90 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf89, primals_228, primals_229, primals_230, primals_231, buf90, 32256, grid=grid(32256), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_232, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf85, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf93 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_142, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf92, primals_238, primals_239, primals_240, primals_241, buf93, 27648, grid=grid(27648), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_242, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf95 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_145, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf94, primals_243, primals_244, primals_245, primals_246, buf95, 27648, grid=grid(27648), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_247, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf97 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_148, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf96, primals_248, primals_249, primals_250, primals_251, buf97, 32256, grid=grid(32256), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_252, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf99 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_151, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf98, primals_253, primals_254, primals_255, primals_256, buf99, 32256, grid=grid(32256), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_257, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf101 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf85, buf101, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf103 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf86, primals_218, primals_219, primals_220, primals_221, buf91, primals_233, primals_234, primals_235, primals_236, buf100, primals_258, primals_259, primals_260, primals_261, buf102, primals_263, primals_264, primals_265, primals_266, buf103, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf103, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf106 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf105, primals_273, primals_274, primals_275, primals_276, buf106, 27648, grid=grid(27648), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_277, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf108 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf107, primals_278, primals_279, primals_280, primals_281, buf108, 32256, grid=grid(32256), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_282, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf103, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf111 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf110, primals_288, primals_289, primals_290, primals_291, buf111, 27648, grid=grid(27648), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_292, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf113 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf112, primals_293, primals_294, primals_295, primals_296, buf113, 27648, grid=grid(27648), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_297, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf115 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_178, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf114, primals_298, primals_299, primals_300, primals_301, buf115, 32256, grid=grid(32256), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_302, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf117 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_181, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf116, primals_303, primals_304, primals_305, primals_306, buf117, 32256, grid=grid(32256), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_307, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf119 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf103, buf119, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf121 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf104, primals_268, primals_269, primals_270, primals_271, buf109, primals_283, primals_284, primals_285, primals_286, buf118, primals_308, primals_309, primals_310, primals_311, buf120, primals_313, primals_314, primals_315, primals_316, buf121, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf121, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf124 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf123, primals_323, primals_324, primals_325, primals_326, buf124, 27648, grid=grid(27648), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_327, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf126 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf125, primals_328, primals_329, primals_330, primals_331, buf126, 32256, grid=grid(32256), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_332, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf121, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf129 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf128, primals_338, primals_339, primals_340, primals_341, buf129, 27648, grid=grid(27648), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_342, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf131 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf130, primals_343, primals_344, primals_345, primals_346, buf131, 27648, grid=grid(27648), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_347, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf133 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf132, primals_348, primals_349, primals_350, primals_351, buf133, 32256, grid=grid(32256), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_352, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf135 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf134, primals_353, primals_354, primals_355, primals_356, buf135, 32256, grid=grid(32256), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_357, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf137 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf121, buf137, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf139 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf122, primals_318, primals_319, primals_320, primals_321, buf127, primals_333, primals_334, primals_335, primals_336, buf136, primals_358, primals_359, primals_360, primals_361, buf138, primals_363, primals_364, primals_365, primals_366, buf139, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf139, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf142 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf141, primals_373, primals_374, primals_375, primals_376, buf142, 27648, grid=grid(27648), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_377, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf144 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf143, primals_378, primals_379, primals_380, primals_381, buf144, 32256, grid=grid(32256), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_382, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf139, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf147 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf146, primals_388, primals_389, primals_390, primals_391, buf147, 27648, grid=grid(27648), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_392, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf149 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf148, primals_393, primals_394, primals_395, primals_396, buf149, 27648, grid=grid(27648), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_397, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf151 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf150, primals_398, primals_399, primals_400, primals_401, buf151, 32256, grid=grid(32256), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_402, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf153 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf152, primals_403, primals_404, primals_405, primals_406, buf153, 32256, grid=grid(32256), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_407, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf155 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf139, buf155, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_412, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf157 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf140, primals_368, primals_369, primals_370, primals_371, buf145, primals_383, primals_384, primals_385, primals_386, buf154, primals_408, primals_409, primals_410, primals_411, buf156, primals_413, primals_414, primals_415, primals_416, buf157, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf157, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf160 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf159, primals_423, primals_424, primals_425, primals_426, buf160, 27648, grid=grid(27648), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_427, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf162 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf161, primals_428, primals_429, primals_430, primals_431, buf162, 32256, grid=grid(32256), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_432, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf157, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf165 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf164, primals_438, primals_439, primals_440, primals_441, buf165, 27648, grid=grid(27648), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_442, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf167 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf166, primals_443, primals_444, primals_445, primals_446, buf167, 27648, grid=grid(27648), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_447, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf169 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf168, primals_448, primals_449, primals_450, primals_451, buf169, 32256, grid=grid(32256), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_452, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf171 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_271, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf170, primals_453, primals_454, primals_455, primals_456, buf171, 32256, grid=grid(32256), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_457, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf173 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf157, buf173, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_276], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf175 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf158, primals_418, primals_419, primals_420, primals_421, buf163, primals_433, primals_434, primals_435, primals_436, buf172, primals_458, primals_459, primals_460, primals_461, buf174, primals_463, primals_464, primals_465, primals_466, buf175, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf175, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf178 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf177, primals_473, primals_474, primals_475, primals_476, buf178, 27648, grid=grid(27648), stream=stream0)
        del primals_476
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_477, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf180 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_286, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf179, primals_478, primals_479, primals_480, primals_481, buf180, 32256, grid=grid(32256), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_482, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf175, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf183 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf182, primals_488, primals_489, primals_490, primals_491, buf183, 27648, grid=grid(27648), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_492, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf185 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_295, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf184, primals_493, primals_494, primals_495, primals_496, buf185, 27648, grid=grid(27648), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_497, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf187 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf186, primals_498, primals_499, primals_500, primals_501, buf187, 32256, grid=grid(32256), stream=stream0)
        del primals_501
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_502, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf189 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf188, primals_503, primals_504, primals_505, primals_506, buf189, 32256, grid=grid(32256), stream=stream0)
        del primals_506
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_507, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf191 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf175, buf191, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf193 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf176, primals_468, primals_469, primals_470, primals_471, buf181, primals_483, primals_484, primals_485, primals_486, buf190, primals_508, primals_509, primals_510, primals_511, buf192, primals_513, primals_514, primals_515, primals_516, buf193, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 384, 6, 6), (13824, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_312], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf193, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf196 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_313, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf195, primals_523, primals_524, primals_525, primals_526, buf196, 27648, grid=grid(27648), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [x_315], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_527, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf198 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_316, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf197, primals_528, primals_529, primals_530, primals_531, buf198, 32256, grid=grid(32256), stream=stream0)
        del primals_531
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_532, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 256, 6, 6), (9216, 36, 6, 1))
        # Topologically Sorted Source Nodes: [x_321], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf193, primals_537, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf201 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf200, primals_538, primals_539, primals_540, primals_541, buf201, 27648, grid=grid(27648), stream=stream0)
        del primals_541
        # Topologically Sorted Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_542, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf203 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_325, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf202, primals_543, primals_544, primals_545, primals_546, buf203, 27648, grid=grid(27648), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_547, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf205 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf204, primals_548, primals_549, primals_550, primals_551, buf205, 32256, grid=grid(32256), stream=stream0)
        del primals_551
        # Topologically Sorted Source Nodes: [x_330], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_552, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 224, 6, 6), (8064, 36, 6, 1))
        buf207 = empty_strided_cuda((4, 224, 6, 6), (8064, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf206, primals_553, primals_554, primals_555, primals_556, buf207, 32256, grid=grid(32256), stream=stream0)
        del primals_556
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_557, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf209 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf193, buf209, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_562, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf211 = empty_strided_cuda((4, 1024, 6, 6), (36864, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf194, primals_518, primals_519, primals_520, primals_521, buf199, primals_533, primals_534, primals_535, primals_536, buf208, primals_558, primals_559, primals_560, primals_561, buf210, primals_563, primals_564, primals_565, primals_566, buf211, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [x_339], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 192, 6, 6), (6912, 36, 6, 1))
        buf213 = empty_strided_cuda((4, 192, 6, 6), (6912, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf212, primals_568, primals_569, primals_570, primals_571, buf213, 27648, grid=grid(27648), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_572, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 192, 2, 2), (768, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_345], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf211, primals_577, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf216 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf215, primals_578, primals_579, primals_580, primals_581, buf216, 36864, grid=grid(36864), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [x_348], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_582, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 256, 6, 6), (9216, 36, 6, 1))
        buf218 = empty_strided_cuda((4, 256, 6, 6), (9216, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_349, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf217, primals_583, primals_584, primals_585, primals_586, buf218, 36864, grid=grid(36864), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [x_351], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_587, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 320, 6, 6), (11520, 36, 6, 1))
        buf220 = empty_strided_cuda((4, 320, 6, 6), (11520, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf219, primals_588, primals_589, primals_590, primals_591, buf220, 46080, grid=grid(46080), stream=stream0)
        del primals_591
        # Topologically Sorted Source Nodes: [x_354], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_592, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 320, 2, 2), (1280, 4, 2, 1))
        buf226 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        buf222 = reinterpret_tensor(buf226, (4, 1024, 2, 2), (6144, 4, 2, 1), 2048)  # alias
        buf223 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_24.run(buf211, buf222, buf223, 16384, grid=grid(16384), stream=stream0)
        buf224 = reinterpret_tensor(buf226, (4, 192, 2, 2), (6144, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf214, primals_573, primals_574, primals_575, primals_576, buf224, 3072, grid=grid(3072), stream=stream0)
        buf225 = reinterpret_tensor(buf226, (4, 320, 2, 2), (6144, 4, 2, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf221, primals_593, primals_594, primals_595, primals_596, buf225, 5120, grid=grid(5120), stream=stream0)
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_597, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, primals_602, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf229 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_361, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf228, primals_603, primals_604, primals_605, primals_606, buf229, 6144, grid=grid(6144), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [x_363], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_607, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf229, primals_612, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf246 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        buf232 = reinterpret_tensor(buf246, (4, 512, 2, 2), (6144, 4, 2, 1), 1024)  # alias
        # Topologically Sorted Source Nodes: [x1_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf230, primals_608, primals_609, primals_610, primals_611, buf231, primals_613, primals_614, primals_615, primals_616, buf232, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_369], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf226, primals_617, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf234 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_370, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf233, primals_618, primals_619, primals_620, primals_621, buf234, 6144, grid=grid(6144), stream=stream0)
        del primals_621
        # Topologically Sorted Source Nodes: [x_372], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_622, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 448, 2, 2), (1792, 4, 2, 1))
        buf236 = empty_strided_cuda((4, 448, 2, 2), (1792, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf235, primals_623, primals_624, primals_625, primals_626, buf236, 7168, grid=grid(7168), stream=stream0)
        del primals_626
        # Topologically Sorted Source Nodes: [x_375], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_627, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf238 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf237, primals_628, primals_629, primals_630, primals_631, buf238, 8192, grid=grid(8192), stream=stream0)
        del primals_631
        # Topologically Sorted Source Nodes: [x_378], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_632, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_381], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf238, primals_637, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf241 = reinterpret_tensor(buf246, (4, 512, 2, 2), (6144, 4, 2, 1), 3072)  # alias
        # Topologically Sorted Source Nodes: [x2_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf239, primals_633, primals_634, primals_635, primals_636, buf240, primals_638, primals_639, primals_640, primals_641, buf241, 8192, grid=grid(8192), stream=stream0)
        buf242 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_31.run(buf226, buf242, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_384], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_642, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf244 = reinterpret_tensor(buf246, (4, 256, 2, 2), (6144, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf227, primals_598, primals_599, primals_600, primals_601, buf244, 4096, grid=grid(4096), stream=stream0)
        buf245 = reinterpret_tensor(buf246, (4, 256, 2, 2), (6144, 4, 2, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [x_385, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf243, primals_643, primals_644, primals_645, primals_646, buf245, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_387], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_647, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf246, primals_652, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf249 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_391, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf248, primals_653, primals_654, primals_655, primals_656, buf249, 6144, grid=grid(6144), stream=stream0)
        del primals_656
        # Topologically Sorted Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_657, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_396], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf249, primals_662, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf266 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        buf252 = reinterpret_tensor(buf266, (4, 512, 2, 2), (6144, 4, 2, 1), 1024)  # alias
        # Topologically Sorted Source Nodes: [x1_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf250, primals_658, primals_659, primals_660, primals_661, buf251, primals_663, primals_664, primals_665, primals_666, buf252, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_399], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf246, primals_667, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf254 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_400, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf253, primals_668, primals_669, primals_670, primals_671, buf254, 6144, grid=grid(6144), stream=stream0)
        del primals_671
        # Topologically Sorted Source Nodes: [x_402], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_672, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 448, 2, 2), (1792, 4, 2, 1))
        buf256 = empty_strided_cuda((4, 448, 2, 2), (1792, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_403, x_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf255, primals_673, primals_674, primals_675, primals_676, buf256, 7168, grid=grid(7168), stream=stream0)
        del primals_676
        # Topologically Sorted Source Nodes: [x_405], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_677, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf258 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_406, x_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf257, primals_678, primals_679, primals_680, primals_681, buf258, 8192, grid=grid(8192), stream=stream0)
        del primals_681
        # Topologically Sorted Source Nodes: [x_408], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_682, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_411], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf258, primals_687, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf261 = reinterpret_tensor(buf266, (4, 512, 2, 2), (6144, 4, 2, 1), 3072)  # alias
        # Topologically Sorted Source Nodes: [x2_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf259, primals_683, primals_684, primals_685, primals_686, buf260, primals_688, primals_689, primals_690, primals_691, buf261, 8192, grid=grid(8192), stream=stream0)
        buf262 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_31.run(buf246, buf262, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_414], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_692, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf264 = reinterpret_tensor(buf266, (4, 256, 2, 2), (6144, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf247, primals_648, primals_649, primals_650, primals_651, buf264, 4096, grid=grid(4096), stream=stream0)
        buf265 = reinterpret_tensor(buf266, (4, 256, 2, 2), (6144, 4, 2, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [x_415, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf263, primals_693, primals_694, primals_695, primals_696, buf265, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_697, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_420], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf266, primals_702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf269 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_421, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf268, primals_703, primals_704, primals_705, primals_706, buf269, 6144, grid=grid(6144), stream=stream0)
        del primals_706
        # Topologically Sorted Source Nodes: [x_423], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_707, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_426], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf269, primals_712, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf286 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        buf272 = reinterpret_tensor(buf286, (4, 512, 2, 2), (6144, 4, 2, 1), 1024)  # alias
        # Topologically Sorted Source Nodes: [x1_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf270, primals_708, primals_709, primals_710, primals_711, buf271, primals_713, primals_714, primals_715, primals_716, buf272, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_429], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf266, primals_717, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 384, 2, 2), (1536, 4, 2, 1))
        buf274 = empty_strided_cuda((4, 384, 2, 2), (1536, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf273, primals_718, primals_719, primals_720, primals_721, buf274, 6144, grid=grid(6144), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_722, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 448, 2, 2), (1792, 4, 2, 1))
        buf276 = empty_strided_cuda((4, 448, 2, 2), (1792, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_433, x_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf275, primals_723, primals_724, primals_725, primals_726, buf276, 7168, grid=grid(7168), stream=stream0)
        del primals_726
        # Topologically Sorted Source Nodes: [x_435], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_727, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf278 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_436, x_437], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf277, primals_728, primals_729, primals_730, primals_731, buf278, 8192, grid=grid(8192), stream=stream0)
        del primals_731
        # Topologically Sorted Source Nodes: [x_438], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_732, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 256, 2, 2), (1024, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_441], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf278, primals_737, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf281 = reinterpret_tensor(buf286, (4, 512, 2, 2), (6144, 4, 2, 1), 3072)  # alias
        # Topologically Sorted Source Nodes: [x2_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf279, primals_733, primals_734, primals_735, primals_736, buf280, primals_738, primals_739, primals_740, primals_741, buf281, 8192, grid=grid(8192), stream=stream0)
        buf282 = empty_strided_cuda((4, 1536, 2, 2), (6144, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_31.run(buf266, buf282, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_444], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_742, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf284 = reinterpret_tensor(buf286, (4, 256, 2, 2), (6144, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_418, x_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf267, primals_698, primals_699, primals_700, primals_701, buf284, 4096, grid=grid(4096), stream=stream0)
        buf285 = reinterpret_tensor(buf286, (4, 256, 2, 2), (6144, 4, 2, 1), 5120)  # alias
        # Topologically Sorted Source Nodes: [x_445, x_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf283, primals_743, primals_744, primals_745, primals_746, buf285, 4096, grid=grid(4096), stream=stream0)
        buf287 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_33.run(buf286, buf287, 6144, grid=grid(6144), stream=stream0)
        del buf272
        del buf281
        del buf284
        del buf285
        del buf286
    return (reinterpret_tensor(buf287, (4, 1536), (1536, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_527, primals_528, primals_529, primals_530, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_627, primals_628, primals_629, primals_630, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_672, primals_673, primals_674, primals_675, primals_677, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, buf0, buf1, buf2, buf3, buf4, buf5, buf7, buf8, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf24, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf82, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf223, buf226, buf227, buf228, buf229, buf230, buf231, buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf240, buf242, buf243, buf246, buf247, buf248, buf249, buf250, buf251, buf253, buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf262, buf263, buf266, buf267, buf268, buf269, buf270, buf271, buf273, buf274, buf275, buf276, buf277, buf278, buf279, buf280, buf282, buf283, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 128, 128), (49152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 64, 1, 7), (448, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 64, 7, 1), (448, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((224, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((256, 256, 1, 7), (1792, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((320, 256, 7, 1), (1792, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((256, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((256, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((448, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((512, 448, 1, 3), (1344, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((256, 512, 1, 3), (1536, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((256, 512, 3, 1), (1536, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((256, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((256, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((448, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((512, 448, 1, 3), (1344, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((256, 512, 1, 3), (1536, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((256, 512, 3, 1), (1536, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((256, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((256, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((448, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((512, 448, 1, 3), (1344, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((256, 512, 1, 3), (1536, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((256, 512, 3, 1), (1536, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
