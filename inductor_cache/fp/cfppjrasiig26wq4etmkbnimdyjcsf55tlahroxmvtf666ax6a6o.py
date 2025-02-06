# AOT ID: ['59_forward']
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


# kernel path: inductor_cache/oc/cocgqe627hqesvtla5o7te3sovt62uofwplksrnopx5fhaeikbah.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_1 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%primals_1,), kwargs = {})
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37mxikslkv2hwe5ja3idotoutne5fedorsq4afewu7eijxmpmmm.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_6 => add_3, mul_4, mul_5, sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfukxdj6cdw66vvx6k5x3cewg6nkhd7euy3aq2iyy6babirh67ok.py
# Topologically Sorted Source Nodes: [x, x_1, x_20], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => relu_2
#   x_1 => constant_pad_nd
#   x_20 => constant_pad_nd_2
# Graph fragment:
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_2, [1, 0, 1, 0], 0.0), kwargs = {})
#   %constant_pad_nd_2 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_3, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_2 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
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
    tl.store(out_ptr1 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwv4s6unp6ymn5jayc4chorkgki2mcmpowoyu22ecmiq34s7a26.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   x_4 => clone
#   x_5 => add_5, mul_7, mul_8, sub_2
#   x_6 => relu_3
# Graph fragment:
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %unsqueeze_206), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x2 = ((xindex // 4) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, xmask)
    tl.store(out_ptr1 + (x5), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dp/cdpqls47qm75hcm7a2d43v45w6qwjldg3mshvx3fotlmxtsrh54l.py
# Topologically Sorted Source Nodes: [input_3, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_3 => add_1, mul_1, mul_2, sub
#   x_10 => relu_4
#   x_11 => constant_pad_nd_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %constant_pad_nd_1 : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_4, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 5) % 5)
    x0 = (xindex % 5)
    x4 = xindex // 25
    x2 = ((xindex // 25) % 4)
    x6 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl.load(in_ptr2 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = 0.001
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp18 = tl.load(in_ptr3 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr4 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp5, tmp23, tmp24)
    tl.store(out_ptr0 + (x6), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w7/cw7kwcmdvm2defk64mj56u47ssdrt5b4r5jawxdxs45jr7iilrte.py
# Topologically Sorted Source Nodes: [x_9, x_19, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_19 => add_11, mul_16, mul_17, sub_5
#   x_9 => add_7, mul_10, mul_11, sub_3
#   x_comb_iter_0 => add_12
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xktn7lq43e76dz6go5djfqecmsdr5t7ejareyr3xywblgkja7p.py
# Topologically Sorted Source Nodes: [x_21, x_34], Original ATen: [aten.max_pool2d_with_indices, aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_21 => _low_memory_max_pool2d_with_offsets, getitem_1
#   x_34 => avg_pool2d
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_2, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd_2, [3, 3], [2, 2], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_max_pool2d_with_indices_6 = async_compile.triton('triton_poi_fused_avg_pool2d_max_pool2d_with_indices_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_max_pool2d_with_indices_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6) + 2*x0 + 10*x1 + 25*x2), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5) + 2*x0 + 10*x1 + 25*x2), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-4) + 2*x0 + 10*x1 + 25*x2), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 10*x1 + 25*x2), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 10*x1 + 25*x2), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 10*x1 + 25*x2), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (4 + 2*x0 + 10*x1 + 25*x2), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5 + 2*x0 + 10*x1 + 25*x2), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6 + 2*x0 + 10*x1 + 25*x2), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
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
    tmp77 = tl.load(in_ptr0 + ((-6) + 2*x0 + 10*x1 + 25*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr0 + ((-5) + 2*x0 + 10*x1 + 25*x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp77
    tmp80 = tl.load(in_ptr0 + ((-4) + 2*x0 + 10*x1 + 25*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp79
    tmp82 = tl.load(in_ptr0 + ((-1) + 2*x0 + 10*x1 + 25*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp82 + tmp81
    tmp84 = tl.load(in_ptr0 + (2*x0 + 10*x1 + 25*x2), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 + tmp83
    tmp86 = tl.load(in_ptr0 + (1 + 2*x0 + 10*x1 + 25*x2), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 + tmp85
    tmp88 = tl.load(in_ptr0 + (4 + 2*x0 + 10*x1 + 25*x2), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp88 + tmp87
    tmp90 = tl.load(in_ptr0 + (5 + 2*x0 + 10*x1 + 25*x2), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp89
    tmp92 = tl.load(in_ptr0 + (6 + 2*x0 + 10*x1 + 25*x2), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp92 + tmp91
    tmp94 = ((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))) + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5))) + ((-1)*((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))) + ((-1)*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))*((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5))))
    tmp95 = tmp93 / tmp94
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp76, xmask)
    tl.store(out_ptr2 + (x4), tmp95, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vk/cvkw2ydlc4cuvngxbcvdk7x2vaify46s6zye72pblsbooa2dimak.py
# Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_46 => relu_10
#   x_47 => constant_pad_nd_6
# Graph fragment:
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
#   %constant_pad_nd_6 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_10, [1, 0, 1, 0], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_7 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
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
    tmp6 = tl.load(in_ptr0 + ((-3) + x0 + 2*x1 + 4*x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7m/c7mxdtdwh2774bblf3hbldbuyb5pxeqftwnrjklbt5qyprebulmh.py
# Topologically Sorted Source Nodes: [x_22, x_32, x_comb_iter_1, x_comb_iter_3_right, x_comb_iter_3, x_46, x_55, x_comb_iter_4], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.add, aten.avg_pool2d, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_22 => clone_2
#   x_32 => add_16, mul_22, mul_23, sub_7
#   x_46 => relu_10
#   x_55 => add_27, mul_34, mul_35, sub_11
#   x_comb_iter_1 => add_17
#   x_comb_iter_3 => add_23
#   x_comb_iter_3_right => avg_pool2d_1
#   x_comb_iter_4 => add_28
# Graph fragment:
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_2, %add_16), kwargs = {})
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_12, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool2d_1, %add_17), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %clone_2), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_10, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x6 = xindex
    x2 = ((xindex // 4) % 4)
    x3 = xindex // 16
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x6), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr6 + (x6), xmask)
    tmp78 = tl.load(in_ptr7 + (x6), xmask)
    tmp79 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = (-1) + x1
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = (-1) + x0
    tmp25 = tmp24 >= tmp19
    tmp26 = tmp24 < tmp21
    tmp27 = tmp25 & tmp26
    tmp28 = tmp23 & tmp27
    tmp29 = tl.load(in_ptr6 + ((-3) + x6), tmp28 & xmask, other=0.0)
    tmp30 = x0
    tmp31 = tmp30 >= tmp19
    tmp32 = tmp30 < tmp21
    tmp33 = tmp31 & tmp32
    tmp34 = tmp23 & tmp33
    tmp35 = tl.load(in_ptr6 + ((-2) + x6), tmp34 & xmask, other=0.0)
    tmp36 = tmp35 + tmp29
    tmp37 = 1 + x0
    tmp38 = tmp37 >= tmp19
    tmp39 = tmp37 < tmp21
    tmp40 = tmp38 & tmp39
    tmp41 = tmp23 & tmp40
    tmp42 = tl.load(in_ptr6 + ((-1) + x6), tmp41 & xmask, other=0.0)
    tmp43 = tmp42 + tmp36
    tmp44 = x1
    tmp45 = tmp44 >= tmp19
    tmp46 = tmp44 < tmp21
    tmp47 = tmp45 & tmp46
    tmp48 = tmp47 & tmp27
    tmp49 = tl.load(in_ptr6 + ((-1) + x6), tmp48 & xmask, other=0.0)
    tmp50 = tmp49 + tmp43
    tmp51 = tmp47 & tmp33
    tmp52 = tl.load(in_ptr6 + (x6), tmp51 & xmask, other=0.0)
    tmp53 = tmp52 + tmp50
    tmp54 = tmp47 & tmp40
    tmp55 = tl.load(in_ptr6 + (1 + x6), tmp54 & xmask, other=0.0)
    tmp56 = tmp55 + tmp53
    tmp57 = 1 + x1
    tmp58 = tmp57 >= tmp19
    tmp59 = tmp57 < tmp21
    tmp60 = tmp58 & tmp59
    tmp61 = tmp60 & tmp27
    tmp62 = tl.load(in_ptr6 + (1 + x6), tmp61 & xmask, other=0.0)
    tmp63 = tmp62 + tmp56
    tmp64 = tmp60 & tmp33
    tmp65 = tl.load(in_ptr6 + (2 + x6), tmp64 & xmask, other=0.0)
    tmp66 = tmp65 + tmp63
    tmp67 = tmp60 & tmp40
    tmp68 = tl.load(in_ptr6 + (3 + x6), tmp67 & xmask, other=0.0)
    tmp69 = tmp68 + tmp66
    tmp70 = 4 + ((-2)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) + ((-2)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp71 = tmp69 / tmp70
    tmp72 = tmp71 + tmp17
    tmp74 = tl.full([1], 0, tl.int32)
    tmp75 = triton_helpers.maximum(tmp74, tmp73)
    tmp76 = 0.0
    tmp77 = tmp75 <= tmp76
    tmp80 = tmp78 - tmp79
    tmp82 = tmp81 + tmp5
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tmp8 / tmp83
    tmp85 = tmp84 * tmp10
    tmp86 = tmp80 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp0
    tl.store(out_ptr0 + (x5 + 64*x3), tmp17, xmask)
    tl.store(out_ptr2 + (x5 + 64*x3), tmp72, xmask)
    tl.store(out_ptr3 + (x6), tmp77, xmask)
    tl.store(out_ptr4 + (x5 + 64*x3), tmp91, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm74pfxsgbcyboax52kaq2czatj2sjryupprw5w5wsmts5ezcptj.py
# Topologically Sorted Source Nodes: [x_35, x_45, x_comb_iter_2], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_35 => clone_4
#   x_45 => add_21, mul_28, mul_29, sub_9
#   x_comb_iter_2 => add_22
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_20,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_4, %add_21), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_clone_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_clone_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_clone_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_clone_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x5 = xindex
    x2 = ((xindex // 4) % 4)
    x3 = xindex // 16
    x6 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x5), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x6 + 64*x3), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_8, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_14, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_20, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_26, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_32, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_38, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, ), (1, ))
    assert_size_stride(primals_43, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_44, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, ), (1, ))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, ), (1, ))
    assert_size_stride(primals_55, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_56, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (4, ), (1, ))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (4, ), (1, ))
    assert_size_stride(primals_61, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_62, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (4, ), (1, ))
    assert_size_stride(primals_67, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_69, (4, ), (1, ))
    assert_size_stride(primals_70, (4, ), (1, ))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(primals_1, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(primals_7, buf2, 256, grid=grid(256), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 16, 4, 1))
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf3, primals_9, primals_10, primals_11, primals_12, buf4, 256, grid=grid(256), stream=stream0)
        buf5 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        buf18 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_20], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_2.run(buf4, buf5, buf18, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_13, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf6, (4, 4, 3, 3), (36, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 3, 3), (36, 9, 3, 1))
        buf8 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf49 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3.run(buf7, primals_15, primals_16, primals_17, primals_18, buf8, buf49, 64, grid=grid(64), stream=stream0)
        del primals_15
        del primals_18
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_19, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf9, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 2, 2), (16, 4, 2, 1))
        buf11 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_constant_pad_nd_relu_4.run(buf1, primals_3, primals_4, primals_5, primals_6, buf11, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_25, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf12, (4, 4, 3, 3), (36, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 3, 3), (36, 9, 3, 1))
        buf14 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf48 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15, x_16], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3.run(buf13, primals_27, primals_28, primals_29, primals_30, buf14, buf48, 64, grid=grid(64), stream=stream0)
        del primals_27
        del primals_30
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_31, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf15, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 4, 2, 2), (16, 4, 2, 1))
        buf17 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_19, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_5.run(buf10, primals_21, primals_22, primals_23, primals_24, buf16, primals_33, primals_34, primals_35, primals_36, buf17, 64, grid=grid(64), stream=stream0)
        del primals_24
        del primals_36
        buf19 = buf13; del buf13  # reuse
        buf20 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.int8)
        buf27 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_21, x_34], Original ATen: [aten.max_pool2d_with_indices, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_max_pool2d_with_indices_6.run(buf18, buf19, buf20, buf27, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf11, primals_37, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf21, (4, 4, 3, 3), (36, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 3, 3), (36, 9, 3, 1))
        buf23 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf47 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28, x_29], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3.run(buf22, primals_39, primals_40, primals_41, primals_42, buf23, buf47, 64, grid=grid(64), stream=stream0)
        del primals_39
        del primals_42
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_43, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf24, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 4, 2, 2), (16, 4, 2, 1))
        buf34 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_7.run(buf17, buf34, 144, grid=grid(144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf35, (4, 4, 3, 3), (36, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 3, 3), (36, 9, 3, 1))
        buf37 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf44 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_50, x_51, x_52], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3.run(buf36, primals_63, primals_64, primals_65, primals_66, buf37, buf44, 64, grid=grid(64), stream=stream0)
        del buf36
        del primals_63
        del primals_66
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf38, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 4, 2, 2), (16, 4, 2, 1))
        buf43 = reinterpret_tensor(buf4, (4, 16, 2, 2), (64, 4, 2, 1), 0); del buf4  # reuse
        buf26 = reinterpret_tensor(buf43, (4, 4, 2, 2), (64, 4, 2, 1), 0)  # alias
        buf41 = reinterpret_tensor(buf43, (4, 4, 2, 2), (64, 4, 2, 1), 32)  # alias
        buf45 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.bool)
        buf42 = reinterpret_tensor(buf43, (4, 4, 2, 2), (64, 4, 2, 1), 48)  # alias
        # Topologically Sorted Source Nodes: [x_22, x_32, x_comb_iter_1, x_comb_iter_3_right, x_comb_iter_3, x_46, x_55, x_comb_iter_4], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.add, aten.avg_pool2d, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8.run(buf19, buf25, primals_45, primals_46, primals_47, primals_48, buf17, buf39, primals_69, primals_70, primals_71, primals_72, buf26, buf41, buf45, buf42, 64, grid=grid(64), stream=stream0)
        del buf19
        del primals_48
        del primals_72
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf11, primals_49, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf28, (4, 4, 3, 3), (36, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 4, 3, 3), (36, 9, 3, 1))
        buf30 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf46 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_40, x_41, x_42], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_native_batch_norm_backward_relu_3.run(buf29, primals_51, primals_52, primals_53, primals_54, buf30, buf46, 64, grid=grid(64), stream=stream0)
        del buf29
        del primals_51
        del primals_54
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_55, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf31, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 4, 2, 2), (16, 4, 2, 1))
        buf40 = reinterpret_tensor(buf43, (4, 4, 2, 2), (64, 4, 2, 1), 16)  # alias
        # Topologically Sorted Source Nodes: [x_35, x_45, x_comb_iter_2], Original ATen: [aten.clone, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_clone_9.run(buf27, buf32, primals_57, primals_58, primals_59, primals_60, buf40, 64, grid=grid(64), stream=stream0)
        del buf27
        del primals_60
    return (buf43, primals_2, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_21, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, buf0, buf1, buf2, buf3, buf5, buf6, buf8, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf18, buf20, buf21, buf23, buf24, buf25, buf28, buf30, buf31, buf32, buf34, buf35, buf37, buf38, buf39, buf44, buf45, buf46, buf47, buf48, buf49, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
