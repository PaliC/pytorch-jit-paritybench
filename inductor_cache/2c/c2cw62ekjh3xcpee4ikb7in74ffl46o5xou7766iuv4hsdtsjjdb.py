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


# kernel path: inductor_cache/4x/c4xqapr7wxnipc6zjb6qe6t3acqhbzkcmc6hpdxgbg3wn2q7luox.py
# Topologically Sorted Source Nodes: [conv2d, batch_norm, xout], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   conv2d => convolution
#   xout => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
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


# kernel path: inductor_cache/2f/c2fk4jqi7pmom6d5dap2tfppey2fpvjmwto2kfz5fs63dqmgy374.py
# Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, xout_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   conv2d_1 => convolution_1
#   xout_1 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
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


# kernel path: inductor_cache/42/c425oxgsfq6ce2ztzjnytwilw7ij7lro4yix6a3tbtjmbdxbwlnv.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_2 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/56/c56p4p3p3hvve72t2shwzla43kd4dfoq2zlvzx6wpjicn6e5qem2.py
# Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, xout_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_7, mul_8, sub_2
#   conv2d_2 => convolution_2
#   xout_2 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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


# kernel path: inductor_cache/rq/crqgvgex6w7bmxvebayl3n5lgsfq4x3wubvgx445a57f5t2oxs4n.py
# Topologically Sorted Source Nodes: [hx_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_1 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_4 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpql2rcbm6dscvgmkq3fpw2br43ccboog3elffzgm7gnzscvz3n2.py
# Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, xout_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_10, mul_11, sub_3
#   conv2d_3 => convolution_3
#   xout_3 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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


# kernel path: inductor_cache/7i/c7i2t3kg7g46abce3ksnc5xfo7cmha5xousvtr6thhrmwfqsa6wa.py
# Topologically Sorted Source Nodes: [hx_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_2 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_6 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/a6/ca6y37ohlqrqu5ckjlcvev2i5tdw7yppefexqyghvk3y6enx3nd3.py
# Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4, xout_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_4 => add_9, mul_13, mul_14, sub_4
#   conv2d_4 => convolution_4
#   xout_4 => relu_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/x2/cx2p27acgo5kx5smx23qrxopmiczmcme3ofyrz7vn75k7awxslxt.py
# Topologically Sorted Source Nodes: [hx_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_3 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3d/c3dem33nbsgsmc4plearigdameniq3in2hg2cr6jhbvxihmhnns7.py
# Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5, xout_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_5 => add_11, mul_16, mul_17, sub_5
#   conv2d_5 => convolution_5
#   xout_5 => relu_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rq/crqyhn2zsusimjnmrckiqljxwvuws4zlejyyflttdvisbx3dkhtz.py
# Topologically Sorted Source Nodes: [hx_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_4 => getitem_8, getitem_9
# Graph fragment:
#   %getitem_8 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 0), kwargs = {})
#   %getitem_9 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_10 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_10(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sx/csxhtesbz2ogyvppyjuxlbyg7am3he6ckboabwz2mmzrs7izkb4g.py
# Topologically Sorted Source Nodes: [conv2d_6, batch_norm_6, xout_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_13, mul_19, mul_20, sub_6
#   conv2d_6 => convolution_6
#   xout_6 => relu_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_38, %primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca5p7bqc5br55vq6hicwocsvqy4gkwgbip5psnj6pjvcdxr24l44.py
# Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_7 => convolution_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_44, %primals_45, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/26/c26ojcu6zvwys7jyaf6rfesgql6k74okb3nc4denfpx47gdxhld3.py
# Topologically Sorted Source Nodes: [hx_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_5 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_7, %relu_6], 1), kwargs = {})
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 64)
    x0 = (xindex % 4)
    x2 = xindex // 256
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
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 64, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-32) + x1) + 128*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/un/cuna7vyprstsomfjtyz3lyd6c5sdh6qlvnw2lsiqk7pia3itjoc4.py
# Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx6up => convert_element_type_19
# Graph fragment:
#   %convert_element_type_19 : [num_users=21] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qo/cqo7uws5rm76qvgadrszd3msrtdhq7gufi2t26gc2f6qd74ohxg6.py
# Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   hx6up => add_19, clamp_max
# Graph fragment:
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1), kwargs = {})
#   %clamp_max : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_19, 1), kwargs = {})
triton_poi_fused_add_clamp_15 = async_compile.triton('triton_poi_fused_add_clamp_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = triton_helpers.minimum(tmp10, tmp9)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b2/cb2irrj3cnm6yxpkeltd55hwtb34yiuggxn4bdcidgmu3ngp7mts.py
# Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   hx6up => add_18, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_18, iota, mul_27, sub_11, sub_9
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_18, 0.5), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 0.5), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_27, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_9, 0.0), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_21), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_11, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjbk33i6ebingetstrjvsimekwfsrd6iydfnbuw332a6lujdnqi6.py
# Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx6up => _unsafe_index, _unsafe_index_1, add_22, mul_29, sub_12
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %convert_element_type_19, %convert_element_type_21]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %convert_element_type_19, %clamp_max_1]), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_2), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_29), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_17 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v5/cv5h35bg4btyja7ytkbckxsbbav4qdlgopfaraxrojcynnzq3bij.py
# Topologically Sorted Source Nodes: [hx_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_6 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_24, %relu_5], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 64)
    x3 = xindex // 1024
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 512*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 2, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 2*tmp10 + 4*(x2) + 128*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 2*tmp10 + 4*(x2) + 128*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 64, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-32) + x2) + 512*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrpcim6lj7ogtva6ahimzvr4tix5dzfbwpp7io4scqxtxvl57qv.py
# Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx5dup => convert_element_type_25
# Graph fragment:
#   %convert_element_type_25 : [num_users=21] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_19 = async_compile.triton('triton_poi_fused__to_copy_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nz/cnzfgqrw6a4jmybmpar6wfdtu5d3edrbmo6hpbozyuzvundcffru.py
# Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   hx5dup => add_28, clamp_max_4
# Graph fragment:
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_25, 1), kwargs = {})
#   %clamp_max_4 : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_28, 3), kwargs = {})
triton_poi_fused_add_clamp_20 = async_compile.triton('triton_poi_fused_add_clamp_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_20(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2b/c2bbdilrti6uhdt6xjiyyncgb53siaizqr3oaym4rp7guj2y4m5r.py
# Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   hx5dup => add_27, clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_24, iota_2, mul_35, sub_17, sub_19
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_24, 0.5), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.5), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_35, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_17, 0.0), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_27), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_19, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=19] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5ygfskfwdvvi2gtk3iglkluamumyechpzikqvje4uhbt6kijvb.py
# Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx5dup => _unsafe_index_4, _unsafe_index_5, add_31, mul_37, sub_20
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %convert_element_type_25, %clamp_max_5]), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %clamp_max_6), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_37), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_22 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/tg/ctg6zl3cgsqqofi6ixe3kjhemlvbnzpsafa6dfutlu4jmiehfp2m.py
# Topologically Sorted Source Nodes: [hx_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_7 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_33, %relu_4], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 64)
    x3 = xindex // 4096
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 2048*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 64, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-32) + x2) + 2048*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cxizjtlckkagu72lgawktdn6hjn2xocvpjhu4zi5ak5c56vfxz35.py
# Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx4dup => convert_element_type_31
# Graph fragment:
#   %convert_element_type_31 : [num_users=17] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_24 = async_compile.triton('triton_poi_fused__to_copy_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuqbtjfsz75jsbmgjcobmw3zxzz2kxc5h3ausb3y6dehjosci3g5.py
# Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   hx4dup => add_37, clamp_max_8
# Graph fragment:
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_31, 1), kwargs = {})
#   %clamp_max_8 : [num_users=15] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_37, 7), kwargs = {})
triton_poi_fused_add_clamp_25 = async_compile.triton('triton_poi_fused_add_clamp_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 7, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuziialfgn6kqbvrrk5fmnx44sktojerjfo3pfkp7mbvhxmcesmt.py
# Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   hx4dup => add_36, clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_30, iota_4, mul_43, sub_25, sub_27
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_30, 0.5), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, 0.5), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_43, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_25, 0.0), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_33), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_27, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=15] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jy/cjybeqwypl5pyuujckhrmq34ccpnwaf7ht55ix5pj4xma5m2gxek.py
# Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx4dup => _unsafe_index_8, _unsafe_index_9, add_40, mul_45, sub_28
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_10, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_10, [None, None, %convert_element_type_31, %clamp_max_9]), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %clamp_max_10), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_45), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_27 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/ws/cwsnrw556mm4zmmtrtxs3teyetdsfo25j2fhbdidovufcb2o4hrz.py
# Topologically Sorted Source Nodes: [hx_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_8 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_42, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_28 = async_compile.triton('triton_poi_fused_cat_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 64)
    x3 = xindex // 16384
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 8192*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 8*tmp10 + 64*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 8*tmp10 + 64*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 64, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 256*((-32) + x2) + 8192*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ty/ctydou3fd6ilhzrm4ki36wbgi3lu4gaifwbhiwtjlzncaxksvymt.py
# Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx3dup => convert_element_type_37
# Graph fragment:
#   %convert_element_type_37 : [num_users=13] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_29 = async_compile.triton('triton_poi_fused__to_copy_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_29(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbruujn6njo35uergazkse6og7ruui4opwvaqwnnmsfkedaornxu.py
# Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   hx3dup => add_46, clamp_max_12
# Graph fragment:
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_37, 1), kwargs = {})
#   %clamp_max_12 : [num_users=11] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_46, 15), kwargs = {})
triton_poi_fused_add_clamp_30 = async_compile.triton('triton_poi_fused_add_clamp_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_30(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 15, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j4/cj4eajmhoedbiofriqknoxpr2argtsnmunusc6gnj6ohpkqzbj7g.py
# Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   hx3dup => add_45, clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_36, iota_6, mul_51, sub_33, sub_35
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_36, 0.5), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, 0.5), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_51, 0.5), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_33, 0.0), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_39), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_35, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=11] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4m/c4mv2jxlnlivm34fmosplfmbb3tm564skllc4s6kancfc4te4whs.py
# Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx3dup => _unsafe_index_12, _unsafe_index_13, add_49, mul_53, sub_36
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_11, [None, None, %convert_element_type_37, %convert_element_type_39]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_11, [None, None, %convert_element_type_37, %clamp_max_13]), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_14), kwargs = {})
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_53), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_32 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/op/copxwrprwa5aqdiuizv34sp3db52zwfp4hp6qwcpd5gn3zaf6zn4.py
# Topologically Sorted Source Nodes: [hx_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_9 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_51, %relu_2], 1), kwargs = {})
triton_poi_fused_cat_33 = async_compile.triton('triton_poi_fused_cat_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 64)
    x3 = xindex // 65536
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 32768*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 16, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 16*tmp10 + 256*(x2) + 8192*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 16*tmp10 + 256*(x2) + 8192*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 64, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 1024*((-32) + x2) + 32768*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/fg/cfgdc2z2mtbpaldnwcozbp2dkvre7q2anyf557evam2zozy6bmgw.py
# Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx2dup => convert_element_type_43
# Graph fragment:
#   %convert_element_type_43 : [num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.int64), kwargs = {})
triton_poi_fused__to_copy_34 = async_compile.triton('triton_poi_fused__to_copy_34', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_34(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q6/cq6lwom4md3zyu4geshyx5ig6czlmh7h2lt3cidzvafu73q5o7ye.py
# Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   hx2dup => add_55, clamp_max_16
# Graph fragment:
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_43, 1), kwargs = {})
#   %clamp_max_16 : [num_users=9] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_55, 31), kwargs = {})
triton_poi_fused_add_clamp_35 = async_compile.triton('triton_poi_fused_add_clamp_35', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_35(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 31, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5mgri5y5aeuhggiygqzxpcua62vfm6hhwf4p64n3gmgaf4prwpf.py
# Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   hx2dup => add_54, clamp_max_18, clamp_min_16, clamp_min_18, convert_element_type_42, iota_8, mul_59, sub_41, sub_43
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_54 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.5), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 0.5), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_59, 0.5), kwargs = {})
#   %clamp_min_16 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_41, 0.0), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_16, %convert_element_type_45), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_43, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=9] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mu/cmuaqm7g4nngrvmdiusvkgzljf24busd2le4djaeyrlckjzcs545.py
# Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx2dup => _unsafe_index_16, _unsafe_index_17, add_58, mul_61, sub_44
# Graph fragment:
#   %_unsafe_index_16 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_12, [None, None, %convert_element_type_43, %convert_element_type_45]), kwargs = {})
#   %_unsafe_index_17 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_12, [None, None, %convert_element_type_43, %clamp_max_17]), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_17, %_unsafe_index_16), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %clamp_max_18), kwargs = {})
#   %add_58 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_16, %mul_61), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_37 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/ds/cdsxrwwvi6mnsbju2wnvisisuwphiarpgakqygueb7n62vx5jc43.py
# Topologically Sorted Source Nodes: [hx_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_10 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_60, %relu_1], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 64)
    x3 = xindex // 262144
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 131072*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 32, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 32*tmp10 + 1024*(x2) + 32768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 32*tmp10 + 1024*(x2) + 32768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 64, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 4096*((-32) + x2) + 131072*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/qh/cqhen6qjmpi5ib2zd5fe5vt6fgzqk5smyoi5ecdn55phz7b2gh54.py
# Topologically Sorted Source Nodes: [conv2d_13, batch_norm_13, xout_13, hx1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_13 => add_62, mul_65, mul_66, sub_48
#   conv2d_13 => convolution_13
#   hx1 => add_63
#   xout_13 => relu_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %primals_80, %primals_81, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_107), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_109), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_62,), kwargs = {})
#   %add_63 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_13, %relu), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ug/cugmea5j7maafgs2azenmswv6rxjpeg2adpif73zyw2b4ao3irhc.py
# Topologically Sorted Source Nodes: [hx_11], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_11 => getitem_10, getitem_11
# Graph fragment:
#   %getitem_10 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_5, 0), kwargs = {})
#   %getitem_11 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_5, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_40 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_40(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccoxhwvyvzj3ek6gx7aapmfp7bixkkjox657jfwg5c3n36tj6ap3.py
# Topologically Sorted Source Nodes: [conv2d_14, batch_norm_14, xout_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_14 => add_65, mul_68, mul_69, sub_49
#   conv2d_14 => convolution_14
#   xout_14 => relu_14
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_86, %primals_87, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_115), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, %unsqueeze_117), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 128)
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


# kernel path: inductor_cache/rn/crnl56tipqkqhhpkzbqfm6bsimqpzzcawhd6aloqnks5ducn6sdy.py
# Topologically Sorted Source Nodes: [conv2d_25, batch_norm_25, xout_25, hx2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_25 => add_115, mul_121, mul_122, sub_88
#   conv2d_25 => convolution_25
#   hx2 => add_116
#   xout_25 => relu_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %primals_152, %primals_153, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_203), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %unsqueeze_205), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %unsqueeze_207), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
#   %add_116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_25, %relu_14), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceiow5ggddlpg4yami6nictrcyx7cj73w6tip4eccsthj4edbrar.py
# Topologically Sorted Source Nodes: [hx_21], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_21 => getitem_20, getitem_21
# Graph fragment:
#   %getitem_20 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_10, 0), kwargs = {})
#   %getitem_21 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_10, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_43 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_43(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqp3vhv645oxczfqmsw7fyrkrhxlpmy55u77k3pnanzzx3axgjjv.py
# Topologically Sorted Source Nodes: [conv2d_26, batch_norm_26, xout_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_26 => add_118, mul_124, mul_125, sub_89
#   conv2d_26 => convolution_26
#   xout_26 => relu_26
# Graph fragment:
#   %convolution_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_20, %primals_158, %primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_211), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_213), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_118,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
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


# kernel path: inductor_cache/c7/cc72hzcz4uy5bp32lgn2qrbtyneghuq6ofuvsqz3isrqmrrkwumq.py
# Topologically Sorted Source Nodes: [conv2d_27, batch_norm_27, xout_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_27 => add_120, mul_127, mul_128, sub_90
#   conv2d_27 => convolution_27
#   xout_27 => relu_27
# Graph fragment:
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %primals_164, %primals_165, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_219), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_221), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_223), kwargs = {})
#   %relu_27 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_120,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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


# kernel path: inductor_cache/re/cre3ls2dcsffso32h4trjaw5f72bvn6d6i3jf5g3p772pjmqbxh3.py
# Topologically Sorted Source Nodes: [hx_22], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_22 => getitem_22, getitem_23
# Graph fragment:
#   %getitem_22 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_11, 0), kwargs = {})
#   %getitem_23 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_11, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_46 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_46(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/kq/ckq7jaedrxa6rmniowjhhprstqd2gzuumawhjbajuur6nn6fjk4u.py
# Topologically Sorted Source Nodes: [conv2d_28, batch_norm_28, xout_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_28 => add_122, mul_130, mul_131, sub_91
#   conv2d_28 => convolution_28
#   xout_28 => relu_28
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_22, %primals_170, %primals_171, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_227), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_229), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_122,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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


# kernel path: inductor_cache/p3/cp3eti5q5742bgjsfaoqqj6lb7eg6lz7nzuvidthi237e3epwlw7.py
# Topologically Sorted Source Nodes: [hx_23], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_23 => getitem_24, getitem_25
# Graph fragment:
#   %getitem_24 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_12, 0), kwargs = {})
#   %getitem_25 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_12, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_48 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_48(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2nxrkdmwvia4r24kwf5dbpivkr65cw7b3jd5b5uw7syvxkxtcf.py
# Topologically Sorted Source Nodes: [conv2d_29, batch_norm_29, xout_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_29 => add_124, mul_133, mul_134, sub_92
#   conv2d_29 => convolution_29
#   xout_29 => relu_29
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_24, %primals_176, %primals_177, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_235), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_237), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_239), kwargs = {})
#   %relu_29 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_124,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/2p/c2pqcrgrwglvo2awvvyibdgbkiqyhpgvaoixxjtlirmfge2dccje.py
# Topologically Sorted Source Nodes: [hx_24], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_24 => getitem_26, getitem_27
# Graph fragment:
#   %getitem_26 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_13, 0), kwargs = {})
#   %getitem_27 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_13, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_50 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_50(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yp/cyp6huy5o5uuyrc33ylbjqx7frl54m6uaogzoslhk5tb6kpopp4i.py
# Topologically Sorted Source Nodes: [conv2d_30, batch_norm_30, xout_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_30 => add_126, mul_136, mul_137, sub_93
#   conv2d_30 => convolution_30
#   xout_30 => relu_30
# Graph fragment:
#   %convolution_30 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_26, %primals_182, %primals_183, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_243), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_245), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_126,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 64)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2l/c2lp4kq4qqfgxk6552hn2vg6mi4gaeeewmdzcbckfmouhhlmzkm4.py
# Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_31 => convolution_31
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_30, %primals_188, %primals_189, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_52 = async_compile.triton('triton_poi_fused_convolution_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_52(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gk/cgk32edptt2gfulcwhgifbml2a3lylz5mw6vuwfppktrxiu7d5f3.py
# Topologically Sorted Source Nodes: [hx_25], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_25 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_31, %relu_30], 1), kwargs = {})
triton_poi_fused_cat_53 = async_compile.triton('triton_poi_fused_cat_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 256*x2), tmp4 & xmask, other=0.0)
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
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-64) + x1) + 256*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yg/cygvzfmdw3wcewwauevcbmnbbtk2mkjgswssyczs5lu5knuab5xm.py
# Topologically Sorted Source Nodes: [hx4dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx4dup_2 => _unsafe_index_36, _unsafe_index_37, add_135, mul_146, sub_99
# Graph fragment:
#   %_unsafe_index_36 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_32, [None, None, %convert_element_type_19, %convert_element_type_21]), kwargs = {})
#   %_unsafe_index_37 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_32, [None, None, %convert_element_type_19, %clamp_max_1]), kwargs = {})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_37, %_unsafe_index_36), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %clamp_max_2), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_36, %mul_146), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_54 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
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


# kernel path: inductor_cache/hv/chvbic7jb2ifvmwvpzriu4kkvyay4kccmtye2atodnxyt5kzh4v2.py
# Topologically Sorted Source Nodes: [hx_26], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_26 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_137, %relu_29], 1), kwargs = {})
triton_poi_fused_cat_55 = async_compile.triton('triton_poi_fused_cat_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 128)
    x3 = xindex // 2048
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 1024*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 2, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 2*tmp10 + 4*(x2) + 256*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 2*tmp10 + 4*(x2) + 256*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 128, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-64) + x2) + 1024*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/st/cstlmyfzhlfu6mpcfct3vmwmo6ennomfzlx6ukv6tahvzirjep3q.py
# Topologically Sorted Source Nodes: [hx3dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx3dup_2 => _unsafe_index_40, _unsafe_index_41, add_144, mul_154, sub_107
# Graph fragment:
#   %_unsafe_index_40 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_33, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_41 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_33, [None, None, %convert_element_type_25, %clamp_max_5]), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_41, %_unsafe_index_40), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %clamp_max_6), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_40, %mul_154), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_56 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/m3/cm3mmmak252fkg5zpqfqx5cb36kvu5ucv7kexwnyd4wlgyhu5tzs.py
# Topologically Sorted Source Nodes: [hx_27], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_27 => cat_13
# Graph fragment:
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_146, %relu_28], 1), kwargs = {})
triton_poi_fused_cat_57 = async_compile.triton('triton_poi_fused_cat_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 128)
    x3 = xindex // 8192
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 4096*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 1024*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 1024*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 128, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-64) + x2) + 4096*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/jm/cjmpyjfjhx5suwkivbsn7xjhsqa7iptxwhhvis52ltz5cs77c7on.py
# Topologically Sorted Source Nodes: [hx2dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx2dup_2 => _unsafe_index_44, _unsafe_index_45, add_153, mul_162, sub_115
# Graph fragment:
#   %_unsafe_index_44 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_34, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_45 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_34, [None, None, %convert_element_type_31, %clamp_max_9]), kwargs = {})
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_45, %_unsafe_index_44), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %clamp_max_10), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_44, %mul_162), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_58 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/55/c556oeq3a7chyv43mq3lfbg7mbmbqvefmdd7cgshpbuttu4lalvo.py
# Topologically Sorted Source Nodes: [hx_28], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_28 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_155, %relu_27], 1), kwargs = {})
triton_poi_fused_cat_59 = async_compile.triton('triton_poi_fused_cat_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 128)
    x3 = xindex // 32768
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 16384*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 8*tmp10 + 64*(x2) + 4096*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 8*tmp10 + 64*(x2) + 4096*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 128, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 256*((-64) + x2) + 16384*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/6d/c6dujgnbqj7tpb7pmorwqieypyrze7humxmzvyz4cpku3pwgknib.py
# Topologically Sorted Source Nodes: [conv2d_35, batch_norm_35, xout_35, hx3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_35 => add_157, mul_166, mul_167, sub_119
#   conv2d_35 => convolution_35
#   hx3 => add_158
#   xout_35 => relu_35
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_14, %primals_212, %primals_213, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_283), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_285), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_287), kwargs = {})
#   %relu_35 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
#   %add_158 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_35, %relu_26), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/hi/chijv4qamrbqi2cj35fuuxqk37afhonjpxkkxb2bkdpngnestrox.py
# Topologically Sorted Source Nodes: [hx_29], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_29 => getitem_28, getitem_29
# Graph fragment:
#   %getitem_28 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_14, 0), kwargs = {})
#   %getitem_29 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_14, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_61 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_61(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/7m/c7mijgvhevchiyyxzked7cubgfkah7fouperi5ostdil22srqpjr.py
# Topologically Sorted Source Nodes: [conv2d_36, batch_norm_36, xout_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_36 => add_160, mul_169, mul_170, sub_120
#   conv2d_36 => convolution_36
#   xout_36 => relu_36
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_28, %primals_218, %primals_219, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_289), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_291), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_293), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_295), kwargs = {})
#   %relu_36 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_160,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_62', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
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


# kernel path: inductor_cache/ij/cijdjffttq2oiludegg7mqc4xqgrpfu636zmnq4sr7l5omu6vvob.py
# Topologically Sorted Source Nodes: [conv2d_37, batch_norm_37, xout_37], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_37 => add_162, mul_172, mul_173, sub_121
#   conv2d_37 => convolution_37
#   xout_37 => relu_37
# Graph fragment:
#   %convolution_37 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_36, %primals_224, %primals_225, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_299), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_301), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_303), kwargs = {})
#   %relu_37 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_162,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
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


# kernel path: inductor_cache/hz/chz6xfupuijpcjbkuxllafraa32taqondkx6phb26wsys36pgjh6.py
# Topologically Sorted Source Nodes: [hx_30], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_30 => getitem_30, getitem_31
# Graph fragment:
#   %getitem_30 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_15, 0), kwargs = {})
#   %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_15, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_64 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_64(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/gx/cgxyl6bb5pvkcqfcyatnvcjkpfwuens4etkj2u2aj4whw4ct7lf6.py
# Topologically Sorted Source Nodes: [conv2d_38, batch_norm_38, xout_38], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_38 => add_164, mul_175, mul_176, sub_122
#   conv2d_38 => convolution_38
#   xout_38 => relu_38
# Graph fragment:
#   %convolution_38 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_30, %primals_230, %primals_231, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_307), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_309), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_311), kwargs = {})
#   %relu_38 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_164,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
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


# kernel path: inductor_cache/ej/cejgklycnnlipp6pdefswqwnm63h7kis27ltorwwmni6xqozcvvl.py
# Topologically Sorted Source Nodes: [hx_31], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_31 => getitem_32, getitem_33
# Graph fragment:
#   %getitem_32 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_16, 0), kwargs = {})
#   %getitem_33 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_16, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_66 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_66(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4vm54ttd3e5stqvldl4gv57iwr2gdwociklrhemcv2qum65kge.py
# Topologically Sorted Source Nodes: [conv2d_39, batch_norm_39, xout_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_39 => add_166, mul_178, mul_179, sub_123
#   conv2d_39 => convolution_39
#   xout_39 => relu_39
# Graph fragment:
#   %convolution_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_32, %primals_236, %primals_237, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_315), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_317), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_166,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 128)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kb/ckbhjqzctr4b2eonhftid6w2fmutbw3rqlwgrxqganm6it2dsl3u.py
# Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_40 => convolution_40
# Graph fragment:
#   %convolution_40 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %primals_242, %primals_243, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_68 = async_compile.triton('triton_poi_fused_convolution_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_68(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ty/ctypbisks23ix3mqorovwvdzpw4kvmgbn4n5hpi3vqnkzbeonfv6.py
# Topologically Sorted Source Nodes: [hx_32], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_32 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_40, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_69 = async_compile.triton('triton_poi_fused_cat_69', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 256)
    x0 = (xindex % 4)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 512*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 256, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-128) + x1) + 512*x2), tmp25, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxvjxv4gymdnj34d24ticgk5or5unvmvudisjktbwyvbulcrnx5f.py
# Topologically Sorted Source Nodes: [hx3dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx3dup_3 => _unsafe_index_48, _unsafe_index_49, add_175, mul_188, sub_129
# Graph fragment:
#   %_unsafe_index_48 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_41, [None, None, %convert_element_type_19, %convert_element_type_21]), kwargs = {})
#   %_unsafe_index_49 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_41, [None, None, %convert_element_type_19, %clamp_max_1]), kwargs = {})
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_49, %_unsafe_index_48), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %clamp_max_2), kwargs = {})
#   %add_175 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_48, %mul_188), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_70 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_70', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
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


# kernel path: inductor_cache/56/c56if2szi2n42vbeqpvbwyie3tgwxgpbiugbmvj5zueaqqsb4ka5.py
# Topologically Sorted Source Nodes: [hx_33], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_33 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_177, %relu_38], 1), kwargs = {})
triton_poi_fused_cat_71 = async_compile.triton('triton_poi_fused_cat_71', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 256)
    x3 = xindex // 4096
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 2048*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 2, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 2*tmp10 + 4*(x2) + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 2*tmp10 + 4*(x2) + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 256, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-128) + x2) + 2048*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxej7pqzm4ylrraml3k6my7mvq5fr4hj67j5nohlf6yuedmbiwp.py
# Topologically Sorted Source Nodes: [hx2dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx2dup_3 => _unsafe_index_52, _unsafe_index_53, add_184, mul_196, sub_137
# Graph fragment:
#   %_unsafe_index_52 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_42, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_53 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_42, [None, None, %convert_element_type_25, %clamp_max_5]), kwargs = {})
#   %sub_137 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_53, %_unsafe_index_52), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_137, %clamp_max_6), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_52, %mul_196), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_72 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_72', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/v6/cv6oaoaifddaqqxzmr7erhx2ugtg5wkoe3c4sgzm35qotfqytcxb.py
# Topologically Sorted Source Nodes: [hx_34], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_34 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_186, %relu_37], 1), kwargs = {})
triton_poi_fused_cat_73 = async_compile.triton('triton_poi_fused_cat_73', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_73(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 256)
    x3 = xindex // 16384
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 8192*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 256, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-128) + x2) + 8192*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwafcgl5gjvxfgr5xhlrxmfku2m4ebp24gy34o67c4ixw6icqyaw.py
# Topologically Sorted Source Nodes: [conv2d_43, batch_norm_43, xout_43, hx4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_43 => add_188, mul_200, mul_201, sub_141
#   conv2d_43 => convolution_43
#   hx4 => add_189
#   xout_43 => relu_43
# Graph fragment:
#   %convolution_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_17, %primals_260, %primals_261, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_141 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_141, %unsqueeze_347), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, %unsqueeze_349), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_201, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_188,), kwargs = {})
#   %add_189 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_43, %relu_36), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_74 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_74', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_74', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_74(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/7p/c7pptjmjyk5smo32cevwy7tqfr55ppji5uu7ybvqv2ixc4akjadz.py
# Topologically Sorted Source Nodes: [hx_35], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_35 => getitem_34, getitem_35
# Graph fragment:
#   %getitem_34 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_17, 0), kwargs = {})
#   %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_17, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_75 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_75', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_75', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_75(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/u7/cu76qkfjafzcrbfq43aj3yed6polqwpmaqvpfawpyblugj2y3rb2.py
# Topologically Sorted Source Nodes: [conv2d_44, batch_norm_44, xout_44], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_44 => add_191, mul_203, mul_204, sub_142
#   conv2d_44 => convolution_44
#   xout_44 => relu_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_34, %primals_266, %primals_267, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_142 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_142, %unsqueeze_355), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %unsqueeze_357), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_204, %unsqueeze_359), kwargs = {})
#   %relu_44 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_191,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
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


# kernel path: inductor_cache/bv/cbvqbvjkajs2qosp5wket3bc7bmpt3tjgsnidmaakwksnpfjre2r.py
# Topologically Sorted Source Nodes: [conv2d_45, batch_norm_45, xout_45], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_45 => add_193, mul_206, mul_207, sub_143
#   conv2d_45 => convolution_45
#   xout_45 => relu_45
# Graph fragment:
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_44, %primals_272, %primals_273, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_143 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_143, %unsqueeze_363), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %unsqueeze_365), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
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


# kernel path: inductor_cache/o5/co5kqqny2xxjk4yzytcata7ago3psojwz4jyao6fyb33mgxdsciy.py
# Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_48 => convolution_48
# Graph fragment:
#   %convolution_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_47, %primals_290, %primals_291, [1, 1], [8, 8], [8, 8], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_78 = async_compile.triton('triton_poi_fused_convolution_78', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_78', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_78(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ah/cahh5d3ds7h2elk3whdvqcnhhhhppajg4vn3xexwa3zw22jh6ws6.py
# Topologically Sorted Source Nodes: [hx_36], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_36 => cat_18
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_48, %relu_47], 1), kwargs = {})
triton_poi_fused_cat_79 = async_compile.triton('triton_poi_fused_cat_79', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 512)
    x0 = (xindex % 16)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 512, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 16*((-256) + x1) + 4096*x2), tmp25, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/zs/czsjf7ie6uqepxjml5psfxbxc44e3dtfviqtdq2bwgqjllcslry5.py
# Topologically Sorted Source Nodes: [conv2d_51, batch_norm_51, xout_51, hx5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_51 => add_205, mul_224, mul_225, sub_149
#   conv2d_51 => convolution_51
#   hx5 => add_206
#   xout_51 => relu_51
# Graph fragment:
#   %convolution_51 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %primals_308, %primals_309, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %unsqueeze_411), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_224, %unsqueeze_413), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_225, %unsqueeze_415), kwargs = {})
#   %relu_51 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_205,), kwargs = {})
#   %add_206 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_51, %relu_44), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mm/cmmxtu7hdkzjxcvlcwy6kz42yt4pcfopxl5466uprdfbswwi3mep.py
# Topologically Sorted Source Nodes: [hx_39], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_39 => getitem_36, getitem_37
# Graph fragment:
#   %getitem_36 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_18, 0), kwargs = {})
#   %getitem_37 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_18, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_81 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_81', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_81(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/2p/c2pcex6exelr7g6qvnlfjsjdru7ppfgmoutxphdzclwg34typ5ma.py
# Topologically Sorted Source Nodes: [conv2d_52, batch_norm_52, xout_52], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_52 => add_208, mul_227, mul_228, sub_150
#   conv2d_52 => convolution_52
#   xout_52 => relu_52
# Graph fragment:
#   %convolution_52 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_36, %primals_314, %primals_315, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_150 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_150, %unsqueeze_419), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_227, %unsqueeze_421), kwargs = {})
#   %add_208 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_228, %unsqueeze_423), kwargs = {})
#   %relu_52 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_208,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_82 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_82', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_82', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_82(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
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


# kernel path: inductor_cache/vl/cvlzrwoawby72ktitaa7ooowinyrloacv2ws42iupzvp5nq67rso.py
# Topologically Sorted Source Nodes: [conv2d_53, batch_norm_53, xout_53], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_53 => add_210, mul_230, mul_231, sub_151
#   conv2d_53 => convolution_53
#   xout_53 => relu_53
# Graph fragment:
#   %convolution_53 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_52, %primals_320, %primals_321, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_151 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_151, %unsqueeze_427), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %unsqueeze_429), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_210,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
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


# kernel path: inductor_cache/sf/csf674dcqbnftdpgc3evy3wuf6r6da6nu4ikuzk6765efi3yvo7j.py
# Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_56 => convolution_56
# Graph fragment:
#   %convolution_56 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_55, %primals_338, %primals_339, [1, 1], [8, 8], [8, 8], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_84 = async_compile.triton('triton_poi_fused_convolution_84', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_84', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_84(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4nsxbgo422bpp2qdqdhk23maj3o4qmwlnikqjh4blgox747fy4.py
# Topologically Sorted Source Nodes: [hx_40], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_40 => cat_21
# Graph fragment:
#   %cat_21 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_56, %relu_55], 1), kwargs = {})
triton_poi_fused_cat_85 = async_compile.triton('triton_poi_fused_cat_85', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_85', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_85(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 512)
    x0 = (xindex % 4)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 1024*x2), tmp4, other=0.0)
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
    tmp26 = tl.full([1], 512, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-256) + x1) + 1024*x2), tmp25, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/fp/cfp3ebdgck7hhxdt2tl24v6j6ajn3okt26oeixx5nuonmfs4im37.py
# Topologically Sorted Source Nodes: [conv2d_59, batch_norm_59, xout_59, hx6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_59 => add_222, mul_248, mul_249, sub_157
#   conv2d_59 => convolution_59
#   hx6 => add_223
#   xout_59 => relu_59
# Graph fragment:
#   %convolution_59 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %primals_356, %primals_357, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_157 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_473), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_157, %unsqueeze_475), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %unsqueeze_477), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_249, %unsqueeze_479), kwargs = {})
#   %relu_59 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_222,), kwargs = {})
#   %add_223 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_59, %relu_52), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_86 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_86', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_86', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_86(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ci/cci74gvhue3purg3eobn33qzfoh644fk5g3yjzo4gdgz3lbhfdhi.py
# Topologically Sorted Source Nodes: [hx6up_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx6up_1 => _unsafe_index_56, _unsafe_index_57, add_228, mul_252, sub_161
# Graph fragment:
#   %_unsafe_index_56 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_223, [None, None, %convert_element_type_19, %convert_element_type_21]), kwargs = {})
#   %_unsafe_index_57 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_223, [None, None, %convert_element_type_19, %clamp_max_1]), kwargs = {})
#   %sub_161 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_57, %_unsafe_index_56), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_161, %clamp_max_2), kwargs = {})
#   %add_228 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_56, %mul_252), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_87 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_87', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_87', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_87(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
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


# kernel path: inductor_cache/s5/cs5gk5twheoyn32mqakicxk4tjhtcadb3vyjfcoahyz5cvgxwp4h.py
# Topologically Sorted Source Nodes: [hx_43], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_43 => cat_24
# Graph fragment:
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_230, %add_206], 1), kwargs = {})
triton_poi_fused_cat_88 = async_compile.triton('triton_poi_fused_cat_88', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_88(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 1024)
    x3 = xindex // 16384
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 8192*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 2, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 2*tmp10 + 4*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 2*tmp10 + 4*(x2) + 2048*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 1024, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-512) + x2) + 8192*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckagjyrjtnwmxukhoktrt2fmva5e3n5ojko3ry775cacluxbzq4j.py
# Topologically Sorted Source Nodes: [hx5dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx5dup_2 => _unsafe_index_60, _unsafe_index_61, add_252, mul_281, sub_176
# Graph fragment:
#   %_unsafe_index_60 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_247, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_61 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_247, [None, None, %convert_element_type_25, %clamp_max_5]), kwargs = {})
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_61, %_unsafe_index_60), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_176, %clamp_max_6), kwargs = {})
#   %add_252 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_60, %mul_281), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_89 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_89', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_89', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_89(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccobqt2s74j6c7ufda7tuy2rufar6o6vzwa3l3a7sx4edlsl2qow.py
# Topologically Sorted Source Nodes: [hx_47], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_47 => cat_28
# Graph fragment:
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_254, %add_189], 1), kwargs = {})
triton_poi_fused_cat_90 = async_compile.triton('triton_poi_fused_cat_90', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_90', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_90(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 1024)
    x3 = xindex // 65536
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 32768*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 8192*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 8192*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 1024, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-512) + x2) + 32768*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwr5l2kq2qgwdumo3bnrvu35h6ti7kw4ionvagqwqg2qfpurvqvp.py
# Topologically Sorted Source Nodes: [conv2d_68, batch_norm_68, xout_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_68 => add_256, mul_285, mul_286, sub_180
#   conv2d_68 => convolution_68
#   xout_68 => relu_68
# Graph fragment:
#   %convolution_68 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_28, %primals_410, %primals_411, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_180 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_180, %unsqueeze_547), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_285, %unsqueeze_549), kwargs = {})
#   %add_256 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_286, %unsqueeze_551), kwargs = {})
#   %relu_68 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_256,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_91 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_91', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_91', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_91(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
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


# kernel path: inductor_cache/5i/c5iazahvqkjbixq4bxckxgzjfgthqcqgbxew5kt67l6awndtq2uo.py
# Topologically Sorted Source Nodes: [conv2d_75, batch_norm_75, xout_75, hx4d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_75 => add_284, mul_316, mul_317, sub_201
#   conv2d_75 => convolution_75
#   hx4d => add_285
#   xout_75 => relu_75
# Graph fragment:
#   %convolution_75 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_31, %primals_452, %primals_453, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_601), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_603), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_605), kwargs = {})
#   %add_284 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_607), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_284,), kwargs = {})
#   %add_285 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_75, %relu_68), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_92 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_92', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_92', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_92(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/e4/ce4kiesygn55xgzhcgtqrmxq3epdeanfu3l53zffevws5fspmm62.py
# Topologically Sorted Source Nodes: [hx4dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx4dup_3 => _unsafe_index_72, _unsafe_index_73, add_290, mul_320, sub_205
# Graph fragment:
#   %_unsafe_index_72 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_285, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_73 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_285, [None, None, %convert_element_type_31, %clamp_max_9]), kwargs = {})
#   %sub_205 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_73, %_unsafe_index_72), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %clamp_max_10), kwargs = {})
#   %add_290 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_72, %mul_320), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_93 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_93', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_93', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_93(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/go/cgod6465qx4srzhd3mzuebdimtej2u53ydwu3jfcfb6d64dwfono.py
# Topologically Sorted Source Nodes: [hx_53], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_53 => cat_32
# Graph fragment:
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_292, %add_158], 1), kwargs = {})
triton_poi_fused_cat_94 = async_compile.triton('triton_poi_fused_cat_94', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_94', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_94(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 512)
    x3 = xindex // 131072
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 8*tmp10 + 64*(x2) + 16384*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 8*tmp10 + 64*(x2) + 16384*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 512, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 256*((-256) + x2) + 65536*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/n2/cn27t6dynlwp2nlsrcuumm2pufp3uo67lxuvvw3osql366dckzek.py
# Topologically Sorted Source Nodes: [conv2d_76, batch_norm_76, xout_76], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_76 => add_294, mul_324, mul_325, sub_209
#   conv2d_76 => convolution_76
#   xout_76 => relu_76
# Graph fragment:
#   %convolution_76 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_32, %primals_458, %primals_459, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_209 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_609), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_611), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_324, %unsqueeze_613), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %unsqueeze_615), kwargs = {})
#   %relu_76 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_294,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_95 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_95', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_95', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_95(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
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


# kernel path: inductor_cache/4w/c4wruz5tecgvonscpujchku2sy342uleukndtn5chi72gfppbsus.py
# Topologically Sorted Source Nodes: [conv2d_85, batch_norm_85, xout_85, hx3d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_85 => add_333, mul_366, mul_367, sub_239
#   conv2d_85 => convolution_85
#   hx3d => add_334
#   xout_85 => relu_85
# Graph fragment:
#   %convolution_85 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_36, %primals_512, %primals_513, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_239 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_239, %unsqueeze_683), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_366, %unsqueeze_685), kwargs = {})
#   %add_333 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %unsqueeze_687), kwargs = {})
#   %relu_85 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_333,), kwargs = {})
#   %add_334 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_85, %relu_76), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_96 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_96', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_96', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_96(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cgggop62u7z3oodbnbxakfhgpbgej7asinj3jun6lqghvb3rqwga.py
# Topologically Sorted Source Nodes: [hx3dup_6], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx3dup_6 => _unsafe_index_88, _unsafe_index_89, add_339, mul_370, sub_243
# Graph fragment:
#   %_unsafe_index_88 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_334, [None, None, %convert_element_type_37, %convert_element_type_39]), kwargs = {})
#   %_unsafe_index_89 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_334, [None, None, %convert_element_type_37, %clamp_max_13]), kwargs = {})
#   %sub_243 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_89, %_unsafe_index_88), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_243, %clamp_max_14), kwargs = {})
#   %add_339 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_88, %mul_370), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_97 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_97', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_97', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_97(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/iy/ciyarn77wcfdhkrgdbeoaqwqckojgamjzzhwbgakfolaqildicmn.py
# Topologically Sorted Source Nodes: [hx_61], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_61 => cat_37
# Graph fragment:
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_341, %add_116], 1), kwargs = {})
triton_poi_fused_cat_98 = async_compile.triton('triton_poi_fused_cat_98', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_98', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_98(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 256)
    x3 = xindex // 262144
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 131072*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 16, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 16*tmp10 + 256*(x2) + 32768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 16*tmp10 + 256*(x2) + 32768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 256, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 1024*((-128) + x2) + 131072*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/yt/cytvufy5zzejuqbkhpwazislovcrbzh37eeeyiaasa45cd7dfjgo.py
# Topologically Sorted Source Nodes: [conv2d_86, batch_norm_86, xout_86], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_86 => add_343, mul_374, mul_375, sub_247
#   conv2d_86 => convolution_86
#   xout_86 => relu_86
# Graph fragment:
#   %convolution_86 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_37, %primals_518, %primals_519, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_247 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_689), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_247, %unsqueeze_691), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_374, %unsqueeze_693), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_375, %unsqueeze_695), kwargs = {})
#   %relu_86 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_343,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_99 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_99', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_99', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_99(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
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


# kernel path: inductor_cache/qc/cqcoxttbdtynmcjxno2t2rk4mej2ztq3lfxxgid6eikptobhruar.py
# Topologically Sorted Source Nodes: [conv2d_97, batch_norm_97, xout_97, hx2d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_97 => add_393, mul_427, mul_428, sub_286
#   conv2d_97 => convolution_97
#   hx2d => add_394
#   xout_97 => relu_97
# Graph fragment:
#   %convolution_97 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_42, %primals_584, %primals_585, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_286 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_97, %unsqueeze_777), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_286, %unsqueeze_779), kwargs = {})
#   %mul_428 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_427, %unsqueeze_781), kwargs = {})
#   %add_393 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_428, %unsqueeze_783), kwargs = {})
#   %relu_97 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_393,), kwargs = {})
#   %add_394 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_97, %relu_86), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_100 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_100', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_100', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_100(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ow/cowyslpgr2fpbostu74u7kkmq2d3azmh7qu6bcfvuadfzvihvxl4.py
# Topologically Sorted Source Nodes: [hx2dup_7], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx2dup_7 => _unsafe_index_108, _unsafe_index_109, add_399, mul_431, sub_290
# Graph fragment:
#   %_unsafe_index_108 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_394, [None, None, %convert_element_type_43, %convert_element_type_45]), kwargs = {})
#   %_unsafe_index_109 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_394, [None, None, %convert_element_type_43, %clamp_max_17]), kwargs = {})
#   %sub_290 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_109, %_unsafe_index_108), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_290, %clamp_max_18), kwargs = {})
#   %add_399 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_108, %mul_431), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_101 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_101', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_101', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_101(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/t4/ct4ejs6dshkbpc4vwcvkbwhkm6z4csru762k4tvfjynecnphyx3v.py
# Topologically Sorted Source Nodes: [hx_71], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_71 => cat_43
# Graph fragment:
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_401, %add_63], 1), kwargs = {})
triton_poi_fused_cat_102 = async_compile.triton('triton_poi_fused_cat_102', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_102', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_102(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 128)
    x3 = xindex // 524288
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 262144*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 32, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 32*tmp10 + 1024*(x2) + 65536*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 32*tmp10 + 1024*(x2) + 65536*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 128, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 4096*((-64) + x2) + 262144*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cik4p5ee6fnvj6yqbvwzwpty7h7phwsmv2tuownket3nrqalcidf.py
# Topologically Sorted Source Nodes: [conv2d_99, batch_norm_99, xout_99], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_99 => add_405, mul_438, mul_439, sub_295
#   conv2d_99 => convolution_99
#   xout_99 => relu_99
# Graph fragment:
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_98, %primals_596, %primals_597, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_295 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_793), kwargs = {})
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_295, %unsqueeze_795), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_438, %unsqueeze_797), kwargs = {})
#   %add_405 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_439, %unsqueeze_799), kwargs = {})
#   %relu_99 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_405,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_103 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_103', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_103', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_103(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 16)
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


# kernel path: inductor_cache/ye/cyelinflf7cc7xalxk75lry756fdl6iyqkk67es5pt25cftwuo7d.py
# Topologically Sorted Source Nodes: [hx_72], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_72 => getitem_56, getitem_57
# Graph fragment:
#   %getitem_56 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_28, 0), kwargs = {})
#   %getitem_57 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_28, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_104 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_104', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_104', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_104(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/na/cnaxh3j6bbtjhtn53o7quemi3tj45fligmrgasp2owq6mvabmfsh.py
# Topologically Sorted Source Nodes: [conv2d_100, batch_norm_100, xout_100], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_100 => add_407, mul_441, mul_442, sub_296
#   conv2d_100 => convolution_100
#   xout_100 => relu_100
# Graph fragment:
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_56, %primals_602, %primals_603, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_296 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_296, %unsqueeze_803), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_441, %unsqueeze_805), kwargs = {})
#   %add_407 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_442, %unsqueeze_807), kwargs = {})
#   %relu_100 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_407,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 16)
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


# kernel path: inductor_cache/jr/cjrkyswnddmtvcyaexqewe4iialf6aw64kpwb3f3ws5simqgz6ru.py
# Topologically Sorted Source Nodes: [hx_73], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_73 => getitem_58, getitem_59
# Graph fragment:
#   %getitem_58 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_29, 0), kwargs = {})
#   %getitem_59 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_29, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_106 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_106', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_106', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_106(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/at/catuqk4h26m3gh3q2qsxzw4tdmhjjdsn3u2ti5gx5rpkucadzhbd.py
# Topologically Sorted Source Nodes: [conv2d_101, batch_norm_101, xout_101], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_101 => add_409, mul_444, mul_445, sub_297
#   conv2d_101 => convolution_101
#   xout_101 => relu_101
# Graph fragment:
#   %convolution_101 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_58, %primals_608, %primals_609, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_297 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_809), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_297, %unsqueeze_811), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, %unsqueeze_813), kwargs = {})
#   %add_409 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_445, %unsqueeze_815), kwargs = {})
#   %relu_101 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_409,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/up/cupk4qonizfl45t2bspnoo5bnuet2n5yxtt4sm5iru55b5s7hcgn.py
# Topologically Sorted Source Nodes: [hx_74], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_74 => getitem_60, getitem_61
# Graph fragment:
#   %getitem_60 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_30, 0), kwargs = {})
#   %getitem_61 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_30, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_108 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_108', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_108', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_108(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/sy/csyxupfr426fwmzxpfx3dk2nqbj66ajehfiqwimlzsxzlrhuhkmd.py
# Topologically Sorted Source Nodes: [conv2d_102, batch_norm_102, xout_102], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_102 => add_411, mul_447, mul_448, sub_298
#   conv2d_102 => convolution_102
#   xout_102 => relu_102
# Graph fragment:
#   %convolution_102 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_60, %primals_614, %primals_615, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_298 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_447 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_298, %unsqueeze_819), kwargs = {})
#   %mul_448 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_447, %unsqueeze_821), kwargs = {})
#   %add_411 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_448, %unsqueeze_823), kwargs = {})
#   %relu_102 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_411,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
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


# kernel path: inductor_cache/zr/czrrw6lpqrs73m7pfefcuovtmkxlqa4fmd2weoetvxmb7dwj4ive.py
# Topologically Sorted Source Nodes: [hx_75], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_75 => getitem_62, getitem_63
# Graph fragment:
#   %getitem_62 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_31, 0), kwargs = {})
#   %getitem_63 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_31, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_110 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_110', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_110', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_110(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwaoktjryzrpz5q3bxphhrru35p6kt5vre2xqgqshwkjiw6lnlk4.py
# Topologically Sorted Source Nodes: [conv2d_103, batch_norm_103, xout_103], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_103 => add_413, mul_450, mul_451, sub_299
#   conv2d_103 => convolution_103
#   xout_103 => relu_103
# Graph fragment:
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_62, %primals_620, %primals_621, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_299 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_450 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_299, %unsqueeze_827), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_450, %unsqueeze_829), kwargs = {})
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_451, %unsqueeze_831), kwargs = {})
#   %relu_103 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_413,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6zqatwyqf4tlle3gymc7dkpfon2oxkyyqshuigxndhx5u5pire.py
# Topologically Sorted Source Nodes: [hx_76], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_76 => getitem_64, getitem_65
# Graph fragment:
#   %getitem_64 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_32, 0), kwargs = {})
#   %getitem_65 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_32, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_112 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_112', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_112', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_112(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/le/cleb2li56pbbz7qt2re5357acomxcs2glv4x6jn5zuybhada36r4.py
# Topologically Sorted Source Nodes: [conv2d_104, batch_norm_104, xout_104], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_104 => add_415, mul_453, mul_454, sub_300
#   conv2d_104 => convolution_104
#   xout_104 => relu_104
# Graph fragment:
#   %convolution_104 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_64, %primals_626, %primals_627, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_300 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_453 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_300, %unsqueeze_835), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_453, %unsqueeze_837), kwargs = {})
#   %add_415 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_454, %unsqueeze_839), kwargs = {})
#   %relu_104 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_415,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t5/ct5zdbittpq3rqo7sxc6qunimj4bulisgwik5okhfl4hpfrgqkus.py
# Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_105 => convolution_105
# Graph fragment:
#   %convolution_105 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_104, %primals_632, %primals_633, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_114 = async_compile.triton('triton_poi_fused_convolution_114', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_114', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_114(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdolp4za4oeylekwdorrocsdgpu43dniei4rstuyzcqu7hzthray.py
# Topologically Sorted Source Nodes: [hx_77], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_77 => cat_44
# Graph fragment:
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_105, %relu_104], 1), kwargs = {})
triton_poi_fused_cat_115 = async_compile.triton('triton_poi_fused_cat_115', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_115', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_115(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 32)
    x0 = (xindex % 4)
    x2 = xindex // 128
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
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
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 32, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 4*((-16) + x1) + 64*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cm/ccmuimlv3gjhkjej35ux3y54a3s24dvsntpg2xke3tggnuksmtdt.py
# Topologically Sorted Source Nodes: [hx6up_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx6up_2 => _unsafe_index_112, _unsafe_index_113, add_424, mul_463, sub_306
# Graph fragment:
#   %_unsafe_index_112 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_106, [None, None, %convert_element_type_19, %convert_element_type_21]), kwargs = {})
#   %_unsafe_index_113 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_106, [None, None, %convert_element_type_19, %clamp_max_1]), kwargs = {})
#   %sub_306 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_113, %_unsafe_index_112), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_306, %clamp_max_2), kwargs = {})
#   %add_424 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_112, %mul_463), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_116 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_116', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_116', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_116(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckm3556hrkkgr7afaopmab2smwaxglxubzqug3nqpgpwjfr7cqz4.py
# Topologically Sorted Source Nodes: [hx_78], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_78 => cat_45
# Graph fragment:
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_426, %relu_103], 1), kwargs = {})
triton_poi_fused_cat_117 = async_compile.triton('triton_poi_fused_cat_117', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_117', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_117(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 32)
    x3 = xindex // 512
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 256*x3), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 2, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 2*tmp10 + 4*(x2) + 64*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 2*tmp10 + 4*(x2) + 64*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 32, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-16) + x2) + 256*x3), tmp31 & xmask, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mb/cmb5p4th4r2boe6gtf4yytfpebgofrgrqb2x3c4zlehpd5ckju6d.py
# Topologically Sorted Source Nodes: [hx5dup_4], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx5dup_4 => _unsafe_index_116, _unsafe_index_117, add_433, mul_471, sub_314
# Graph fragment:
#   %_unsafe_index_116 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_107, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_117 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_107, [None, None, %convert_element_type_25, %clamp_max_5]), kwargs = {})
#   %sub_314 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_117, %_unsafe_index_116), kwargs = {})
#   %mul_471 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_314, %clamp_max_6), kwargs = {})
#   %add_433 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_116, %mul_471), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_118 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_118', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_118', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_118(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/4q/c4qgsmx6m4q77pilpbkuisayjwsb5jbdljyz35ycs674fq2vsach.py
# Topologically Sorted Source Nodes: [hx_79], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_79 => cat_46
# Graph fragment:
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_435, %relu_102], 1), kwargs = {})
triton_poi_fused_cat_119 = async_compile.triton('triton_poi_fused_cat_119', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_119', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_119(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 32)
    x3 = xindex // 2048
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 1024*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 256*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 256*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 32, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-16) + x2) + 1024*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxmbqspnosv7bud2tq4s7cl77p6pdr5iykt563ce3ogned6ehbx.py
# Topologically Sorted Source Nodes: [hx4dup_6], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx4dup_6 => _unsafe_index_120, _unsafe_index_121, add_442, mul_479, sub_322
# Graph fragment:
#   %_unsafe_index_120 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_108, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_121 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_108, [None, None, %convert_element_type_31, %clamp_max_9]), kwargs = {})
#   %sub_322 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_121, %_unsafe_index_120), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_322, %clamp_max_10), kwargs = {})
#   %add_442 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_120, %mul_479), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_120 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_120', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_120', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_120(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/bz/cbzbfrd4epvf4mk4zvdfkc36h2cd6hdzgol7fkvflkgxssv4lwtp.py
# Topologically Sorted Source Nodes: [hx_80], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_80 => cat_47
# Graph fragment:
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_444, %relu_101], 1), kwargs = {})
triton_poi_fused_cat_121 = async_compile.triton('triton_poi_fused_cat_121', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_121', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_121(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 32)
    x3 = xindex // 8192
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 4096*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 8*tmp10 + 64*(x2) + 1024*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 8*tmp10 + 64*(x2) + 1024*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 32, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 256*((-16) + x2) + 4096*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/au/cauocbdnuox3ducmpbcqowm6wy3s7ku6myosy4elwgmf4zs7gxny.py
# Topologically Sorted Source Nodes: [hx3dup_8], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx3dup_8 => _unsafe_index_124, _unsafe_index_125, add_451, mul_487, sub_330
# Graph fragment:
#   %_unsafe_index_124 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_109, [None, None, %convert_element_type_37, %convert_element_type_39]), kwargs = {})
#   %_unsafe_index_125 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_109, [None, None, %convert_element_type_37, %clamp_max_13]), kwargs = {})
#   %sub_330 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_125, %_unsafe_index_124), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_330, %clamp_max_14), kwargs = {})
#   %add_451 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_124, %mul_487), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_122 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_122', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_122', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_122(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/hh/chhtmmj5qurqnurydutxrakzz7ieaszvskm37tajn6hcjl7jepgj.py
# Topologically Sorted Source Nodes: [hx_81], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_81 => cat_48
# Graph fragment:
#   %cat_48 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_453, %relu_100], 1), kwargs = {})
triton_poi_fused_cat_123 = async_compile.triton('triton_poi_fused_cat_123', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_123', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_123(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 32)
    x3 = xindex // 32768
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 16384*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 16, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 16*tmp10 + 256*(x2) + 4096*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 16*tmp10 + 256*(x2) + 4096*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 32, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 1024*((-16) + x2) + 16384*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/a6/ca67yzevj5zucipgdvqzj65mzb6hg55avzoe7pudg4dskvaqdygi.py
# Topologically Sorted Source Nodes: [hx2dup_8], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hx2dup_8 => _unsafe_index_128, _unsafe_index_129, add_460, mul_495, sub_338
# Graph fragment:
#   %_unsafe_index_128 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_110, [None, None, %convert_element_type_43, %convert_element_type_45]), kwargs = {})
#   %_unsafe_index_129 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_110, [None, None, %convert_element_type_43, %clamp_max_17]), kwargs = {})
#   %sub_338 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_129, %_unsafe_index_128), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_338, %clamp_max_18), kwargs = {})
#   %add_460 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_128, %mul_495), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_124 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_124', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_124', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_124(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/2e/c2epbbal6bqk5uavwlryufhinu2jam7zfqdnonqx6mvy7nqvs4n4.py
# Topologically Sorted Source Nodes: [hx_82], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_82 => cat_49
# Graph fragment:
#   %cat_49 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_462, %relu_99], 1), kwargs = {})
triton_poi_fused_cat_125 = async_compile.triton('triton_poi_fused_cat_125', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_125', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_125(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 32)
    x3 = xindex // 131072
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 32, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 32*tmp10 + 1024*(x2) + 16384*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 32*tmp10 + 1024*(x2) + 16384*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 32, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 4096*((-16) + x2) + 65536*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/zi/czij5467kpi2cw57gffaif4kgp5vm3bz74publrcu6jsby57ufyj.py
# Topologically Sorted Source Nodes: [d2, d2_1, sigmoid_2], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   d2 => convolution_113
#   d2_1 => _unsafe_index_132, _unsafe_index_133, _unsafe_index_134, _unsafe_index_135, add_470, add_471, add_472, mul_503, mul_504, mul_505, sub_346, sub_347, sub_349
#   sigmoid_2 => sigmoid_2
# Graph fragment:
#   %convolution_113 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_394, %primals_676, %primals_677, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_132 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_113, [None, None, %convert_element_type_43, %convert_element_type_45]), kwargs = {})
#   %_unsafe_index_133 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_113, [None, None, %convert_element_type_43, %clamp_max_17]), kwargs = {})
#   %_unsafe_index_134 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_113, [None, None, %clamp_max_16, %convert_element_type_45]), kwargs = {})
#   %_unsafe_index_135 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_113, [None, None, %clamp_max_16, %clamp_max_17]), kwargs = {})
#   %sub_346 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_133, %_unsafe_index_132), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_346, %clamp_max_18), kwargs = {})
#   %add_470 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_132, %mul_503), kwargs = {})
#   %sub_347 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_135, %_unsafe_index_134), kwargs = {})
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_347, %clamp_max_18), kwargs = {})
#   %add_471 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_134, %mul_504), kwargs = {})
#   %sub_349 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_471, %add_470), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_349, %clamp_max_19), kwargs = {})
#   %add_472 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_470, %mul_505), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_472,), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_126 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_126', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_126', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_126(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tl.load(in_ptr2 + (tmp16 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp18 = tmp17 + tmp11
    tmp19 = tmp18 - tmp12
    tmp21 = tmp19 * tmp20
    tmp22 = tmp12 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr2 + (tmp8 + 32*tmp26 + 1024*x2), None, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp11
    tmp29 = tl.load(in_ptr2 + (tmp16 + 32*tmp26 + 1024*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30 - tmp28
    tmp32 = tmp31 * tmp20
    tmp33 = tmp28 + tmp32
    tmp34 = tmp33 - tmp22
    tmp36 = tmp34 * tmp35
    tmp37 = tmp22 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/nx/cnxm7el4qspgyjvtduv5sqvjhyyp5vze6x4tghcpuddg5p2sr64v.py
# Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   d3_1 => convert_element_type_361
# Graph fragment:
#   %convert_element_type_361 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_68, torch.int64), kwargs = {})
triton_poi_fused__to_copy_127 = async_compile.triton('triton_poi_fused__to_copy_127', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_127', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_127(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgbr2lnbvuksqglpvmgazbnm6mz335fnqdxsdzjkercbjur3qjb.py
# Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   d3_1 => add_474, clamp_max_136
# Graph fragment:
#   %add_474 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_361, 1), kwargs = {})
#   %clamp_max_136 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_474, 15), kwargs = {})
triton_poi_fused_add_clamp_128 = async_compile.triton('triton_poi_fused_add_clamp_128', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_128', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_128(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 15, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pn/cpn3mygmuzwdnx2s7nyv4auplfleohwzn7u47qh37dsdvbuqspac.py
# Topologically Sorted Source Nodes: [hx2dup, d3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   d3_1 => clamp_max_138, clamp_min_136, clamp_min_138, mul_506, sub_350, sub_352
#   hx2dup => add_54, convert_element_type_42, iota_8
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_54 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.5), kwargs = {})
#   %mul_506 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 0.25), kwargs = {})
#   %sub_350 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_506, 0.5), kwargs = {})
#   %clamp_min_136 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_350, 0.0), kwargs = {})
#   %sub_352 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_136, %convert_element_type_363), kwargs = {})
#   %clamp_min_138 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_352, 0.0), kwargs = {})
#   %clamp_max_138 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_138, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
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


# kernel path: inductor_cache/fy/cfytcnbpxvu5xuhow73wyjcx4ayagbm4ah6zbcu7vm2nd5chvgmo.py
# Topologically Sorted Source Nodes: [d3, d3_1, sigmoid_3], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   d3 => convolution_114
#   d3_1 => _unsafe_index_136, _unsafe_index_137, _unsafe_index_138, _unsafe_index_139, add_477, add_478, add_479, mul_508, mul_509, mul_510, sub_353, sub_354, sub_356
#   sigmoid_3 => sigmoid_3
# Graph fragment:
#   %convolution_114 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_334, %primals_678, %primals_679, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_136 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_114, [None, None, %convert_element_type_361, %convert_element_type_363]), kwargs = {})
#   %_unsafe_index_137 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_114, [None, None, %convert_element_type_361, %clamp_max_137]), kwargs = {})
#   %_unsafe_index_138 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_114, [None, None, %clamp_max_136, %convert_element_type_363]), kwargs = {})
#   %_unsafe_index_139 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_114, [None, None, %clamp_max_136, %clamp_max_137]), kwargs = {})
#   %sub_353 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_137, %_unsafe_index_136), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_353, %clamp_max_138), kwargs = {})
#   %add_477 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_136, %mul_508), kwargs = {})
#   %sub_354 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_139, %_unsafe_index_138), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_354, %clamp_max_138), kwargs = {})
#   %add_478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_138, %mul_509), kwargs = {})
#   %sub_356 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_478, %add_477), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_356, %clamp_max_139), kwargs = {})
#   %add_479 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_477, %mul_510), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_479,), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_130 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_130', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_130', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_130(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tl.load(in_ptr2 + (tmp16 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp18 = tmp17 + tmp11
    tmp19 = tmp18 - tmp12
    tmp21 = tmp19 * tmp20
    tmp22 = tmp12 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr2 + (tmp8 + 16*tmp26 + 256*x2), None, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp11
    tmp29 = tl.load(in_ptr2 + (tmp16 + 16*tmp26 + 256*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30 - tmp28
    tmp32 = tmp31 * tmp20
    tmp33 = tmp28 + tmp32
    tmp34 = tmp33 - tmp22
    tmp36 = tmp34 * tmp35
    tmp37 = tmp22 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4r2fy3b757bpbjvvrvi5oelkydw66yqubcsmaazbc4rc6gsb64l.py
# Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   d4_1 => convert_element_type_365
# Graph fragment:
#   %convert_element_type_365 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_70, torch.int64), kwargs = {})
triton_poi_fused__to_copy_131 = async_compile.triton('triton_poi_fused__to_copy_131', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_131', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_131(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c374cytdtphhifx23pthwfetgely65tyn5yjb4mxdokqugu66pwh.py
# Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   d4_1 => add_481, clamp_max_140
# Graph fragment:
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_365, 1), kwargs = {})
#   %clamp_max_140 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_481, 7), kwargs = {})
triton_poi_fused_add_clamp_132 = async_compile.triton('triton_poi_fused_add_clamp_132', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_132', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_132(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 7, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfuakjvfogdnbrbfe3tvohyvntswmhv3m5wstu75vpapiqglhggi.py
# Topologically Sorted Source Nodes: [hx2dup, d4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   d4_1 => clamp_max_142, clamp_min_140, clamp_min_142, mul_511, sub_357, sub_359
#   hx2dup => add_54, convert_element_type_42, iota_8
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_54 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.5), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 0.125), kwargs = {})
#   %sub_357 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_511, 0.5), kwargs = {})
#   %clamp_min_140 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_357, 0.0), kwargs = {})
#   %sub_359 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_140, %convert_element_type_367), kwargs = {})
#   %clamp_min_142 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_359, 0.0), kwargs = {})
#   %clamp_max_142 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_142, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
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


# kernel path: inductor_cache/dq/cdqvlfuxzblku76g7uuqnwwb2aonk3kq5pkhvpjj5l6mfwgrofyi.py
# Topologically Sorted Source Nodes: [d4, d4_1, sigmoid_4], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   d4 => convolution_115
#   d4_1 => _unsafe_index_140, _unsafe_index_141, _unsafe_index_142, _unsafe_index_143, add_484, add_485, add_486, mul_513, mul_514, mul_515, sub_360, sub_361, sub_363
#   sigmoid_4 => sigmoid_4
# Graph fragment:
#   %convolution_115 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_285, %primals_680, %primals_681, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_140 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_115, [None, None, %convert_element_type_365, %convert_element_type_367]), kwargs = {})
#   %_unsafe_index_141 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_115, [None, None, %convert_element_type_365, %clamp_max_141]), kwargs = {})
#   %_unsafe_index_142 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_115, [None, None, %clamp_max_140, %convert_element_type_367]), kwargs = {})
#   %_unsafe_index_143 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_115, [None, None, %clamp_max_140, %clamp_max_141]), kwargs = {})
#   %sub_360 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_141, %_unsafe_index_140), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_360, %clamp_max_142), kwargs = {})
#   %add_484 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_140, %mul_513), kwargs = {})
#   %sub_361 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_143, %_unsafe_index_142), kwargs = {})
#   %mul_514 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_361, %clamp_max_142), kwargs = {})
#   %add_485 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_142, %mul_514), kwargs = {})
#   %sub_363 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_485, %add_484), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_363, %clamp_max_143), kwargs = {})
#   %add_486 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_484, %mul_515), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_486,), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_134 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_134', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_134', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_134(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tl.load(in_ptr2 + (tmp16 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp18 = tmp17 + tmp11
    tmp19 = tmp18 - tmp12
    tmp21 = tmp19 * tmp20
    tmp22 = tmp12 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr2 + (tmp8 + 8*tmp26 + 64*x2), None, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp11
    tmp29 = tl.load(in_ptr2 + (tmp16 + 8*tmp26 + 64*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30 - tmp28
    tmp32 = tmp31 * tmp20
    tmp33 = tmp28 + tmp32
    tmp34 = tmp33 - tmp22
    tmp36 = tmp34 * tmp35
    tmp37 = tmp22 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/ps/cpsgdbwu75xqnoxnfe6dxf5e6nr4gicceym7ni53gor5qvekjc3i.py
# Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   d5_1 => convert_element_type_369
# Graph fragment:
#   %convert_element_type_369 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_72, torch.int64), kwargs = {})
triton_poi_fused__to_copy_135 = async_compile.triton('triton_poi_fused__to_copy_135', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_135', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_135(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ow/cow3nhxhx6wegtdhvozut4baqj7mp2j6ccwkkyl34qzoq5u6kitk.py
# Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   d5_1 => add_488, clamp_max_144
# Graph fragment:
#   %add_488 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_369, 1), kwargs = {})
#   %clamp_max_144 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_488, 3), kwargs = {})
triton_poi_fused_add_clamp_136 = async_compile.triton('triton_poi_fused_add_clamp_136', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_136', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_136(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtb7lbtadw75slkudqey7ybldhk37sf5dkqsc7cr3n2lkjb6wrx.py
# Topologically Sorted Source Nodes: [hx2dup, d5_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   d5_1 => clamp_max_146, clamp_min_144, clamp_min_146, mul_516, sub_364, sub_366
#   hx2dup => add_54, convert_element_type_42, iota_8
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_54 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.5), kwargs = {})
#   %mul_516 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 0.0625), kwargs = {})
#   %sub_364 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_516, 0.5), kwargs = {})
#   %clamp_min_144 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_364, 0.0), kwargs = {})
#   %sub_366 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_144, %convert_element_type_371), kwargs = {})
#   %clamp_min_146 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_366, 0.0), kwargs = {})
#   %clamp_max_146 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_146, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
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


# kernel path: inductor_cache/na/cnadnzpi5qp4wgha74k27mumhoe7yt37jkpc3euhglfqjm57zbiu.py
# Topologically Sorted Source Nodes: [d5, d5_1, sigmoid_5], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   d5 => convolution_116
#   d5_1 => _unsafe_index_144, _unsafe_index_145, _unsafe_index_146, _unsafe_index_147, add_491, add_492, add_493, mul_518, mul_519, mul_520, sub_367, sub_368, sub_370
#   sigmoid_5 => sigmoid_5
# Graph fragment:
#   %convolution_116 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_247, %primals_682, %primals_683, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_144 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_116, [None, None, %convert_element_type_369, %convert_element_type_371]), kwargs = {})
#   %_unsafe_index_145 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_116, [None, None, %convert_element_type_369, %clamp_max_145]), kwargs = {})
#   %_unsafe_index_146 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_116, [None, None, %clamp_max_144, %convert_element_type_371]), kwargs = {})
#   %_unsafe_index_147 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_116, [None, None, %clamp_max_144, %clamp_max_145]), kwargs = {})
#   %sub_367 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_145, %_unsafe_index_144), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_367, %clamp_max_146), kwargs = {})
#   %add_491 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_144, %mul_518), kwargs = {})
#   %sub_368 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_147, %_unsafe_index_146), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_368, %clamp_max_146), kwargs = {})
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_146, %mul_519), kwargs = {})
#   %sub_370 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_492, %add_491), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_370, %clamp_max_147), kwargs = {})
#   %add_493 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_491, %mul_520), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_493,), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_138 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_138', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_138', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_138(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tl.load(in_ptr2 + (tmp16 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp18 = tmp17 + tmp11
    tmp19 = tmp18 - tmp12
    tmp21 = tmp19 * tmp20
    tmp22 = tmp12 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr2 + (tmp8 + 4*tmp26 + 16*x2), None, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp11
    tmp29 = tl.load(in_ptr2 + (tmp16 + 4*tmp26 + 16*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30 - tmp28
    tmp32 = tmp31 * tmp20
    tmp33 = tmp28 + tmp32
    tmp34 = tmp33 - tmp22
    tmp36 = tmp34 * tmp35
    tmp37 = tmp22 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/vf/cvfcp7xtlblgc7yjndpb5jdnz5ow2bsfre564fveqzgkii6d75ls.py
# Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   d6_1 => convert_element_type_373
# Graph fragment:
#   %convert_element_type_373 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_74, torch.int64), kwargs = {})
triton_poi_fused__to_copy_139 = async_compile.triton('triton_poi_fused__to_copy_139', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_139', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_139(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v5/cv54dste2ijpp3kin3lm4vmeoyfanhmu2uaqepof6xmilfqojqf2.py
# Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   d6_1 => add_495, clamp_max_148
# Graph fragment:
#   %add_495 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_373, 1), kwargs = {})
#   %clamp_max_148 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_495, 1), kwargs = {})
triton_poi_fused_add_clamp_140 = async_compile.triton('triton_poi_fused_add_clamp_140', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_140', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_140(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = triton_helpers.minimum(tmp11, tmp10)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ri/criegothdqvbs2em6p6jwy3l5lmcchue7lewufthea5h6iwcxijq.py
# Topologically Sorted Source Nodes: [hx2dup, d6_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   d6_1 => clamp_max_150, clamp_min_148, clamp_min_150, mul_521, sub_371, sub_373
#   hx2dup => add_54, convert_element_type_42, iota_8
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_54 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.5), kwargs = {})
#   %mul_521 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 0.03125), kwargs = {})
#   %sub_371 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_521, 0.5), kwargs = {})
#   %clamp_min_148 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_371, 0.0), kwargs = {})
#   %sub_373 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_148, %convert_element_type_375), kwargs = {})
#   %clamp_min_150 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_373, 0.0), kwargs = {})
#   %clamp_max_150 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_150, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03125
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


# kernel path: inductor_cache/gd/cgdsrgsdmuluh2h2itbwwb5svtqyoozjps3axl7rcuuevrmwsxxq.py
# Topologically Sorted Source Nodes: [d6, d6_1, sigmoid_6], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   d6 => convolution_117
#   d6_1 => _unsafe_index_148, _unsafe_index_149, _unsafe_index_150, _unsafe_index_151, add_498, add_499, add_500, mul_523, mul_524, mul_525, sub_374, sub_375, sub_377
#   sigmoid_6 => sigmoid_6
# Graph fragment:
#   %convolution_117 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_223, %primals_684, %primals_685, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_148 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_117, [None, None, %convert_element_type_373, %convert_element_type_375]), kwargs = {})
#   %_unsafe_index_149 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_117, [None, None, %convert_element_type_373, %clamp_max_149]), kwargs = {})
#   %_unsafe_index_150 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_117, [None, None, %clamp_max_148, %convert_element_type_375]), kwargs = {})
#   %_unsafe_index_151 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_117, [None, None, %clamp_max_148, %clamp_max_149]), kwargs = {})
#   %sub_374 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_149, %_unsafe_index_148), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_374, %clamp_max_150), kwargs = {})
#   %add_498 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_148, %mul_523), kwargs = {})
#   %sub_375 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_151, %_unsafe_index_150), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_375, %clamp_max_150), kwargs = {})
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_150, %mul_524), kwargs = {})
#   %sub_377 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_499, %add_498), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_377, %clamp_max_151), kwargs = {})
#   %add_500 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_498, %mul_525), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_500,), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_142 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_142', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_142', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_142(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tl.load(in_ptr2 + (tmp16 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp18 = tmp17 + tmp11
    tmp19 = tmp18 - tmp12
    tmp21 = tmp19 * tmp20
    tmp22 = tmp12 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr2 + (tmp8 + 2*tmp26 + 4*x2), None, eviction_policy='evict_last')
    tmp28 = tmp27 + tmp11
    tmp29 = tl.load(in_ptr2 + (tmp16 + 2*tmp26 + 4*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30 - tmp28
    tmp32 = tmp31 * tmp20
    tmp33 = tmp28 + tmp32
    tmp34 = tmp33 - tmp22
    tmp36 = tmp34 * tmp35
    tmp37 = tmp22 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmn6ukgqexhvpesia2wbhwypul5uhsnm5mucipdtos56jy5nxpm.py
# Topologically Sorted Source Nodes: [cat_50], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_50 => cat_50
# Graph fragment:
#   %cat_50 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_112, %add_472, %add_479, %add_486, %add_493, %add_500], 1), kwargs = {})
triton_poi_fused_cat_143 = async_compile.triton('triton_poi_fused_cat_143', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_143', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_143(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 6)
    x0 = (xindex % 4096)
    x2 = xindex // 24576
    x3 = xindex
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (x0 + 4096*x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp20 = tmp0 >= tmp12
    tmp21 = tl.full([1], 3, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 4096*x2), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + 4096*x2), tmp23, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tmp0 >= tmp21
    tmp30 = tl.full([1], 4, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr6 + (x0 + 4096*x2), tmp32, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr7 + (x0 + 4096*x2), tmp32, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp32, tmp35, tmp36)
    tmp38 = tmp0 >= tmp30
    tmp39 = tl.full([1], 5, tl.int64)
    tmp40 = tmp0 < tmp39
    tmp41 = tmp38 & tmp40
    tmp42 = tl.load(in_ptr8 + (x0 + 4096*x2), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr9 + (x0 + 4096*x2), tmp41, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tmp0 >= tmp39
    tmp48 = tl.full([1], 6, tl.int64)
    tmp49 = tmp0 < tmp48
    tmp50 = tl.load(in_ptr10 + (x0 + 4096*x2), tmp47, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr11 + (x0 + 4096*x2), tmp47, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp47, tmp52, tmp53)
    tmp55 = tl.where(tmp41, tmp46, tmp54)
    tmp56 = tl.where(tmp32, tmp37, tmp55)
    tmp57 = tl.where(tmp23, tmp28, tmp56)
    tmp58 = tl.where(tmp14, tmp19, tmp57)
    tmp59 = tl.where(tmp4, tmp10, tmp58)
    tl.store(out_ptr0 + (x3), tmp59, None)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csmrnerluatqexptrbsrexw4k7geh6zwhfw3titoz7c2sdt4hr47.py
# Topologically Sorted Source Nodes: [d0, sigmoid], Original ATen: [aten.convolution, aten.sigmoid]
# Source node to ATen node mapping:
#   d0 => convolution_118
#   sigmoid => sigmoid
# Graph fragment:
#   %convolution_118 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_50, %primals_686, %primals_687, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_118,), kwargs = {})
triton_poi_fused_convolution_sigmoid_144 = async_compile.triton('triton_poi_fused_convolution_sigmoid_144', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_sigmoid_144', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_sigmoid_144(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_33, (32, ), (1, ))
    assert_size_stride(primals_34, (32, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_39, (32, ), (1, ))
    assert_size_stride(primals_40, (32, ), (1, ))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_42, (32, ), (1, ))
    assert_size_stride(primals_43, (32, ), (1, ))
    assert_size_stride(primals_44, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (32, ), (1, ))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_54, (32, ), (1, ))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_57, (32, ), (1, ))
    assert_size_stride(primals_58, (32, ), (1, ))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, ), (1, ))
    assert_size_stride(primals_67, (32, ), (1, ))
    assert_size_stride(primals_68, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_69, (32, ), (1, ))
    assert_size_stride(primals_70, (32, ), (1, ))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (32, ), (1, ))
    assert_size_stride(primals_73, (32, ), (1, ))
    assert_size_stride(primals_74, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (32, ), (1, ))
    assert_size_stride(primals_78, (32, ), (1, ))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, ), (1, ))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, ), (1, ))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (32, ), (1, ))
    assert_size_stride(primals_114, (32, ), (1, ))
    assert_size_stride(primals_115, (32, ), (1, ))
    assert_size_stride(primals_116, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (32, ), (1, ))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (32, ), (1, ))
    assert_size_stride(primals_127, (32, ), (1, ))
    assert_size_stride(primals_128, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (32, ), (1, ))
    assert_size_stride(primals_131, (32, ), (1, ))
    assert_size_stride(primals_132, (32, ), (1, ))
    assert_size_stride(primals_133, (32, ), (1, ))
    assert_size_stride(primals_134, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (32, ), (1, ))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_141, (32, ), (1, ))
    assert_size_stride(primals_142, (32, ), (1, ))
    assert_size_stride(primals_143, (32, ), (1, ))
    assert_size_stride(primals_144, (32, ), (1, ))
    assert_size_stride(primals_145, (32, ), (1, ))
    assert_size_stride(primals_146, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_147, (32, ), (1, ))
    assert_size_stride(primals_148, (32, ), (1, ))
    assert_size_stride(primals_149, (32, ), (1, ))
    assert_size_stride(primals_150, (32, ), (1, ))
    assert_size_stride(primals_151, (32, ), (1, ))
    assert_size_stride(primals_152, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (256, ), (1, ))
    assert_size_stride(primals_162, (256, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (64, ), (1, ))
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_183, (64, ), (1, ))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, ), (1, ))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_201, (64, ), (1, ))
    assert_size_stride(primals_202, (64, ), (1, ))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_207, (64, ), (1, ))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_218, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (128, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (128, ), (1, ))
    assert_size_stride(primals_241, (128, ), (1, ))
    assert_size_stride(primals_242, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_243, (128, ), (1, ))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_249, (128, ), (1, ))
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_261, (512, ), (1, ))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_279, (256, ), (1, ))
    assert_size_stride(primals_280, (256, ), (1, ))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (256, ), (1, ))
    assert_size_stride(primals_288, (256, ), (1, ))
    assert_size_stride(primals_289, (256, ), (1, ))
    assert_size_stride(primals_290, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_291, (256, ), (1, ))
    assert_size_stride(primals_292, (256, ), (1, ))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_294, (256, ), (1, ))
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_297, (256, ), (1, ))
    assert_size_stride(primals_298, (256, ), (1, ))
    assert_size_stride(primals_299, (256, ), (1, ))
    assert_size_stride(primals_300, (256, ), (1, ))
    assert_size_stride(primals_301, (256, ), (1, ))
    assert_size_stride(primals_302, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_304, (256, ), (1, ))
    assert_size_stride(primals_305, (256, ), (1, ))
    assert_size_stride(primals_306, (256, ), (1, ))
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, ), (1, ))
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (512, ), (1, ))
    assert_size_stride(primals_320, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_321, (256, ), (1, ))
    assert_size_stride(primals_322, (256, ), (1, ))
    assert_size_stride(primals_323, (256, ), (1, ))
    assert_size_stride(primals_324, (256, ), (1, ))
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_327, (256, ), (1, ))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (256, ), (1, ))
    assert_size_stride(primals_330, (256, ), (1, ))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_333, (256, ), (1, ))
    assert_size_stride(primals_334, (256, ), (1, ))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_336, (256, ), (1, ))
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_339, (256, ), (1, ))
    assert_size_stride(primals_340, (256, ), (1, ))
    assert_size_stride(primals_341, (256, ), (1, ))
    assert_size_stride(primals_342, (256, ), (1, ))
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_345, (256, ), (1, ))
    assert_size_stride(primals_346, (256, ), (1, ))
    assert_size_stride(primals_347, (256, ), (1, ))
    assert_size_stride(primals_348, (256, ), (1, ))
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_351, (256, ), (1, ))
    assert_size_stride(primals_352, (256, ), (1, ))
    assert_size_stride(primals_353, (256, ), (1, ))
    assert_size_stride(primals_354, (256, ), (1, ))
    assert_size_stride(primals_355, (256, ), (1, ))
    assert_size_stride(primals_356, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_357, (512, ), (1, ))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (512, ), (1, ))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (512, ), (1, ))
    assert_size_stride(primals_362, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_363, (512, ), (1, ))
    assert_size_stride(primals_364, (512, ), (1, ))
    assert_size_stride(primals_365, (512, ), (1, ))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (512, ), (1, ))
    assert_size_stride(primals_368, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_369, (256, ), (1, ))
    assert_size_stride(primals_370, (256, ), (1, ))
    assert_size_stride(primals_371, (256, ), (1, ))
    assert_size_stride(primals_372, (256, ), (1, ))
    assert_size_stride(primals_373, (256, ), (1, ))
    assert_size_stride(primals_374, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_375, (256, ), (1, ))
    assert_size_stride(primals_376, (256, ), (1, ))
    assert_size_stride(primals_377, (256, ), (1, ))
    assert_size_stride(primals_378, (256, ), (1, ))
    assert_size_stride(primals_379, (256, ), (1, ))
    assert_size_stride(primals_380, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_381, (256, ), (1, ))
    assert_size_stride(primals_382, (256, ), (1, ))
    assert_size_stride(primals_383, (256, ), (1, ))
    assert_size_stride(primals_384, (256, ), (1, ))
    assert_size_stride(primals_385, (256, ), (1, ))
    assert_size_stride(primals_386, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_387, (256, ), (1, ))
    assert_size_stride(primals_388, (256, ), (1, ))
    assert_size_stride(primals_389, (256, ), (1, ))
    assert_size_stride(primals_390, (256, ), (1, ))
    assert_size_stride(primals_391, (256, ), (1, ))
    assert_size_stride(primals_392, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_393, (256, ), (1, ))
    assert_size_stride(primals_394, (256, ), (1, ))
    assert_size_stride(primals_395, (256, ), (1, ))
    assert_size_stride(primals_396, (256, ), (1, ))
    assert_size_stride(primals_397, (256, ), (1, ))
    assert_size_stride(primals_398, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_399, (256, ), (1, ))
    assert_size_stride(primals_400, (256, ), (1, ))
    assert_size_stride(primals_401, (256, ), (1, ))
    assert_size_stride(primals_402, (256, ), (1, ))
    assert_size_stride(primals_403, (256, ), (1, ))
    assert_size_stride(primals_404, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (512, ), (1, ))
    assert_size_stride(primals_408, (512, ), (1, ))
    assert_size_stride(primals_409, (512, ), (1, ))
    assert_size_stride(primals_410, (256, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_411, (256, ), (1, ))
    assert_size_stride(primals_412, (256, ), (1, ))
    assert_size_stride(primals_413, (256, ), (1, ))
    assert_size_stride(primals_414, (256, ), (1, ))
    assert_size_stride(primals_415, (256, ), (1, ))
    assert_size_stride(primals_416, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (128, ), (1, ))
    assert_size_stride(primals_420, (128, ), (1, ))
    assert_size_stride(primals_421, (128, ), (1, ))
    assert_size_stride(primals_422, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, ), (1, ))
    assert_size_stride(primals_425, (128, ), (1, ))
    assert_size_stride(primals_426, (128, ), (1, ))
    assert_size_stride(primals_427, (128, ), (1, ))
    assert_size_stride(primals_428, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (128, ), (1, ))
    assert_size_stride(primals_437, (128, ), (1, ))
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (128, ), (1, ))
    assert_size_stride(primals_443, (128, ), (1, ))
    assert_size_stride(primals_444, (128, ), (1, ))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, ), (1, ))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_453, (256, ), (1, ))
    assert_size_stride(primals_454, (256, ), (1, ))
    assert_size_stride(primals_455, (256, ), (1, ))
    assert_size_stride(primals_456, (256, ), (1, ))
    assert_size_stride(primals_457, (256, ), (1, ))
    assert_size_stride(primals_458, (128, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (128, ), (1, ))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_465, (64, ), (1, ))
    assert_size_stride(primals_466, (64, ), (1, ))
    assert_size_stride(primals_467, (64, ), (1, ))
    assert_size_stride(primals_468, (64, ), (1, ))
    assert_size_stride(primals_469, (64, ), (1, ))
    assert_size_stride(primals_470, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_471, (64, ), (1, ))
    assert_size_stride(primals_472, (64, ), (1, ))
    assert_size_stride(primals_473, (64, ), (1, ))
    assert_size_stride(primals_474, (64, ), (1, ))
    assert_size_stride(primals_475, (64, ), (1, ))
    assert_size_stride(primals_476, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_477, (64, ), (1, ))
    assert_size_stride(primals_478, (64, ), (1, ))
    assert_size_stride(primals_479, (64, ), (1, ))
    assert_size_stride(primals_480, (64, ), (1, ))
    assert_size_stride(primals_481, (64, ), (1, ))
    assert_size_stride(primals_482, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_483, (64, ), (1, ))
    assert_size_stride(primals_484, (64, ), (1, ))
    assert_size_stride(primals_485, (64, ), (1, ))
    assert_size_stride(primals_486, (64, ), (1, ))
    assert_size_stride(primals_487, (64, ), (1, ))
    assert_size_stride(primals_488, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_489, (64, ), (1, ))
    assert_size_stride(primals_490, (64, ), (1, ))
    assert_size_stride(primals_491, (64, ), (1, ))
    assert_size_stride(primals_492, (64, ), (1, ))
    assert_size_stride(primals_493, (64, ), (1, ))
    assert_size_stride(primals_494, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_495, (64, ), (1, ))
    assert_size_stride(primals_496, (64, ), (1, ))
    assert_size_stride(primals_497, (64, ), (1, ))
    assert_size_stride(primals_498, (64, ), (1, ))
    assert_size_stride(primals_499, (64, ), (1, ))
    assert_size_stride(primals_500, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_501, (64, ), (1, ))
    assert_size_stride(primals_502, (64, ), (1, ))
    assert_size_stride(primals_503, (64, ), (1, ))
    assert_size_stride(primals_504, (64, ), (1, ))
    assert_size_stride(primals_505, (64, ), (1, ))
    assert_size_stride(primals_506, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_507, (64, ), (1, ))
    assert_size_stride(primals_508, (64, ), (1, ))
    assert_size_stride(primals_509, (64, ), (1, ))
    assert_size_stride(primals_510, (64, ), (1, ))
    assert_size_stride(primals_511, (64, ), (1, ))
    assert_size_stride(primals_512, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_513, (128, ), (1, ))
    assert_size_stride(primals_514, (128, ), (1, ))
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, ), (1, ))
    assert_size_stride(primals_517, (128, ), (1, ))
    assert_size_stride(primals_518, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_519, (64, ), (1, ))
    assert_size_stride(primals_520, (64, ), (1, ))
    assert_size_stride(primals_521, (64, ), (1, ))
    assert_size_stride(primals_522, (64, ), (1, ))
    assert_size_stride(primals_523, (64, ), (1, ))
    assert_size_stride(primals_524, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_525, (32, ), (1, ))
    assert_size_stride(primals_526, (32, ), (1, ))
    assert_size_stride(primals_527, (32, ), (1, ))
    assert_size_stride(primals_528, (32, ), (1, ))
    assert_size_stride(primals_529, (32, ), (1, ))
    assert_size_stride(primals_530, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_531, (32, ), (1, ))
    assert_size_stride(primals_532, (32, ), (1, ))
    assert_size_stride(primals_533, (32, ), (1, ))
    assert_size_stride(primals_534, (32, ), (1, ))
    assert_size_stride(primals_535, (32, ), (1, ))
    assert_size_stride(primals_536, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_537, (32, ), (1, ))
    assert_size_stride(primals_538, (32, ), (1, ))
    assert_size_stride(primals_539, (32, ), (1, ))
    assert_size_stride(primals_540, (32, ), (1, ))
    assert_size_stride(primals_541, (32, ), (1, ))
    assert_size_stride(primals_542, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_543, (32, ), (1, ))
    assert_size_stride(primals_544, (32, ), (1, ))
    assert_size_stride(primals_545, (32, ), (1, ))
    assert_size_stride(primals_546, (32, ), (1, ))
    assert_size_stride(primals_547, (32, ), (1, ))
    assert_size_stride(primals_548, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_549, (32, ), (1, ))
    assert_size_stride(primals_550, (32, ), (1, ))
    assert_size_stride(primals_551, (32, ), (1, ))
    assert_size_stride(primals_552, (32, ), (1, ))
    assert_size_stride(primals_553, (32, ), (1, ))
    assert_size_stride(primals_554, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_555, (32, ), (1, ))
    assert_size_stride(primals_556, (32, ), (1, ))
    assert_size_stride(primals_557, (32, ), (1, ))
    assert_size_stride(primals_558, (32, ), (1, ))
    assert_size_stride(primals_559, (32, ), (1, ))
    assert_size_stride(primals_560, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_561, (32, ), (1, ))
    assert_size_stride(primals_562, (32, ), (1, ))
    assert_size_stride(primals_563, (32, ), (1, ))
    assert_size_stride(primals_564, (32, ), (1, ))
    assert_size_stride(primals_565, (32, ), (1, ))
    assert_size_stride(primals_566, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_567, (32, ), (1, ))
    assert_size_stride(primals_568, (32, ), (1, ))
    assert_size_stride(primals_569, (32, ), (1, ))
    assert_size_stride(primals_570, (32, ), (1, ))
    assert_size_stride(primals_571, (32, ), (1, ))
    assert_size_stride(primals_572, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_573, (32, ), (1, ))
    assert_size_stride(primals_574, (32, ), (1, ))
    assert_size_stride(primals_575, (32, ), (1, ))
    assert_size_stride(primals_576, (32, ), (1, ))
    assert_size_stride(primals_577, (32, ), (1, ))
    assert_size_stride(primals_578, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_579, (32, ), (1, ))
    assert_size_stride(primals_580, (32, ), (1, ))
    assert_size_stride(primals_581, (32, ), (1, ))
    assert_size_stride(primals_582, (32, ), (1, ))
    assert_size_stride(primals_583, (32, ), (1, ))
    assert_size_stride(primals_584, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_585, (64, ), (1, ))
    assert_size_stride(primals_586, (64, ), (1, ))
    assert_size_stride(primals_587, (64, ), (1, ))
    assert_size_stride(primals_588, (64, ), (1, ))
    assert_size_stride(primals_589, (64, ), (1, ))
    assert_size_stride(primals_590, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_591, (64, ), (1, ))
    assert_size_stride(primals_592, (64, ), (1, ))
    assert_size_stride(primals_593, (64, ), (1, ))
    assert_size_stride(primals_594, (64, ), (1, ))
    assert_size_stride(primals_595, (64, ), (1, ))
    assert_size_stride(primals_596, (16, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_597, (16, ), (1, ))
    assert_size_stride(primals_598, (16, ), (1, ))
    assert_size_stride(primals_599, (16, ), (1, ))
    assert_size_stride(primals_600, (16, ), (1, ))
    assert_size_stride(primals_601, (16, ), (1, ))
    assert_size_stride(primals_602, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_603, (16, ), (1, ))
    assert_size_stride(primals_604, (16, ), (1, ))
    assert_size_stride(primals_605, (16, ), (1, ))
    assert_size_stride(primals_606, (16, ), (1, ))
    assert_size_stride(primals_607, (16, ), (1, ))
    assert_size_stride(primals_608, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_609, (16, ), (1, ))
    assert_size_stride(primals_610, (16, ), (1, ))
    assert_size_stride(primals_611, (16, ), (1, ))
    assert_size_stride(primals_612, (16, ), (1, ))
    assert_size_stride(primals_613, (16, ), (1, ))
    assert_size_stride(primals_614, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_615, (16, ), (1, ))
    assert_size_stride(primals_616, (16, ), (1, ))
    assert_size_stride(primals_617, (16, ), (1, ))
    assert_size_stride(primals_618, (16, ), (1, ))
    assert_size_stride(primals_619, (16, ), (1, ))
    assert_size_stride(primals_620, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_621, (16, ), (1, ))
    assert_size_stride(primals_622, (16, ), (1, ))
    assert_size_stride(primals_623, (16, ), (1, ))
    assert_size_stride(primals_624, (16, ), (1, ))
    assert_size_stride(primals_625, (16, ), (1, ))
    assert_size_stride(primals_626, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_627, (16, ), (1, ))
    assert_size_stride(primals_628, (16, ), (1, ))
    assert_size_stride(primals_629, (16, ), (1, ))
    assert_size_stride(primals_630, (16, ), (1, ))
    assert_size_stride(primals_631, (16, ), (1, ))
    assert_size_stride(primals_632, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_633, (16, ), (1, ))
    assert_size_stride(primals_634, (16, ), (1, ))
    assert_size_stride(primals_635, (16, ), (1, ))
    assert_size_stride(primals_636, (16, ), (1, ))
    assert_size_stride(primals_637, (16, ), (1, ))
    assert_size_stride(primals_638, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_639, (16, ), (1, ))
    assert_size_stride(primals_640, (16, ), (1, ))
    assert_size_stride(primals_641, (16, ), (1, ))
    assert_size_stride(primals_642, (16, ), (1, ))
    assert_size_stride(primals_643, (16, ), (1, ))
    assert_size_stride(primals_644, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_645, (16, ), (1, ))
    assert_size_stride(primals_646, (16, ), (1, ))
    assert_size_stride(primals_647, (16, ), (1, ))
    assert_size_stride(primals_648, (16, ), (1, ))
    assert_size_stride(primals_649, (16, ), (1, ))
    assert_size_stride(primals_650, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_651, (16, ), (1, ))
    assert_size_stride(primals_652, (16, ), (1, ))
    assert_size_stride(primals_653, (16, ), (1, ))
    assert_size_stride(primals_654, (16, ), (1, ))
    assert_size_stride(primals_655, (16, ), (1, ))
    assert_size_stride(primals_656, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_657, (16, ), (1, ))
    assert_size_stride(primals_658, (16, ), (1, ))
    assert_size_stride(primals_659, (16, ), (1, ))
    assert_size_stride(primals_660, (16, ), (1, ))
    assert_size_stride(primals_661, (16, ), (1, ))
    assert_size_stride(primals_662, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_663, (16, ), (1, ))
    assert_size_stride(primals_664, (16, ), (1, ))
    assert_size_stride(primals_665, (16, ), (1, ))
    assert_size_stride(primals_666, (16, ), (1, ))
    assert_size_stride(primals_667, (16, ), (1, ))
    assert_size_stride(primals_668, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_669, (64, ), (1, ))
    assert_size_stride(primals_670, (64, ), (1, ))
    assert_size_stride(primals_671, (64, ), (1, ))
    assert_size_stride(primals_672, (64, ), (1, ))
    assert_size_stride(primals_673, (64, ), (1, ))
    assert_size_stride(primals_674, (1, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_675, (1, ), (1, ))
    assert_size_stride(primals_676, (1, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_677, (1, ), (1, ))
    assert_size_stride(primals_678, (1, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_679, (1, ), (1, ))
    assert_size_stride(primals_680, (1, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_681, (1, ), (1, ))
    assert_size_stride(primals_682, (1, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_683, (1, ), (1, ))
    assert_size_stride(primals_684, (1, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_685, (1, ), (1, ))
    assert_size_stride(primals_686, (1, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_687, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d, batch_norm, xout], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, 1048576, grid=grid(1048576), stream=stream0)
        del primals_3
        del primals_7
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, xout_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 524288, grid=grid(524288), stream=stream0)
        del primals_13
        del primals_9
        buf6 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_2.run(buf5, buf6, buf7, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, xout_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf9, primals_15, primals_16, primals_17, primals_18, primals_19, buf10, 131072, grid=grid(131072), stream=stream0)
        del primals_15
        del primals_19
        buf11 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_4.run(buf10, buf11, buf12, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf11, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, xout_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf14, primals_21, primals_22, primals_23, primals_24, primals_25, buf15, 32768, grid=grid(32768), stream=stream0)
        del primals_21
        del primals_25
        buf16 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf15, buf16, buf17, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf16, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4, xout_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf19, primals_27, primals_28, primals_29, primals_30, primals_31, buf20, 8192, grid=grid(8192), stream=stream0)
        del primals_27
        del primals_31
        buf21 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf20, buf21, buf22, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf21, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 32, 4, 4), (512, 16, 4, 1))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5, xout_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf24, primals_33, primals_34, primals_35, primals_36, primals_37, buf25, 2048, grid=grid(2048), stream=stream0)
        del primals_33
        del primals_37
        buf26 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        buf27 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_10.run(buf25, buf26, buf27, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 2, 2), (128, 4, 2, 1))
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_6, batch_norm_6, xout_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf29, primals_39, primals_40, primals_41, primals_42, primals_43, buf30, 512, grid=grid(512), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_44, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 32, 2, 2), (128, 4, 2, 1))
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf32, primals_45, 512, grid=grid(512), stream=stream0)
        del primals_45
        buf33 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf32, primals_46, primals_47, primals_48, primals_49, buf30, buf33, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 32, 2, 2), (128, 4, 2, 1))
        buf35 = buf34; del buf34  # reuse
        buf36 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_8, batch_norm_8, xout_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf35, primals_51, primals_52, primals_53, primals_54, primals_55, buf36, 512, grid=grid(512), stream=stream0)
        del primals_51
        buf37 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf37, 4, grid=grid(4), stream=stream0)
        buf38 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf38, 4, grid=grid(4), stream=stream0)
        buf39 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf39, 4, grid=grid(4), stream=stream0)
        buf40 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf40, 4, grid=grid(4), stream=stream0)
        buf41 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16.run(buf41, 4, grid=grid(4), stream=stream0)
        buf42 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_17.run(buf37, buf39, buf36, buf40, buf41, buf42, 2048, grid=grid(2048), stream=stream0)
        buf43 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx6up], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16.run(buf43, 4, grid=grid(4), stream=stream0)
        buf44 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf42, buf38, buf39, buf36, buf40, buf41, buf43, buf25, buf44, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 32, 4, 4), (512, 16, 4, 1))
        buf46 = buf45; del buf45  # reuse
        buf47 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9, batch_norm_9, xout_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf46, primals_57, primals_58, primals_59, primals_60, primals_61, buf47, 2048, grid=grid(2048), stream=stream0)
        del primals_57
        buf48 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(buf48, 8, grid=grid(8), stream=stream0)
        buf49 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_20.run(buf49, 8, grid=grid(8), stream=stream0)
        buf50 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(buf50, 8, grid=grid(8), stream=stream0)
        buf51 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_20.run(buf51, 8, grid=grid(8), stream=stream0)
        buf52 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21.run(buf52, 8, grid=grid(8), stream=stream0)
        buf53 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_22.run(buf48, buf50, buf47, buf51, buf52, buf53, 8192, grid=grid(8192), stream=stream0)
        buf54 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_21.run(buf54, 8, grid=grid(8), stream=stream0)
        buf55 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf53, buf49, buf50, buf47, buf51, buf52, buf54, buf20, buf55, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf57 = buf56; del buf56  # reuse
        buf58 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [conv2d_10, batch_norm_10, xout_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf57, primals_63, primals_64, primals_65, primals_66, primals_67, buf58, 8192, grid=grid(8192), stream=stream0)
        del primals_63
        buf59 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf59, 16, grid=grid(16), stream=stream0)
        buf60 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_25.run(buf60, 16, grid=grid(16), stream=stream0)
        buf61 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf61, 16, grid=grid(16), stream=stream0)
        buf62 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_25.run(buf62, 16, grid=grid(16), stream=stream0)
        buf63 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26.run(buf63, 16, grid=grid(16), stream=stream0)
        buf64 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_27.run(buf59, buf61, buf58, buf62, buf63, buf64, 32768, grid=grid(32768), stream=stream0)
        buf65 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_26.run(buf65, 16, grid=grid(16), stream=stream0)
        buf66 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf64, buf60, buf61, buf58, buf62, buf63, buf65, buf15, buf66, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf68 = buf67; del buf67  # reuse
        buf69 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11, xout_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf68, primals_69, primals_70, primals_71, primals_72, primals_73, buf69, 32768, grid=grid(32768), stream=stream0)
        del primals_69
        buf70 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf70, 32, grid=grid(32), stream=stream0)
        buf71 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_30.run(buf71, 32, grid=grid(32), stream=stream0)
        buf72 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf72, 32, grid=grid(32), stream=stream0)
        buf73 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_30.run(buf73, 32, grid=grid(32), stream=stream0)
        buf74 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31.run(buf74, 32, grid=grid(32), stream=stream0)
        buf75 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_32.run(buf70, buf72, buf69, buf73, buf74, buf75, 131072, grid=grid(131072), stream=stream0)
        buf76 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_31.run(buf76, 32, grid=grid(32), stream=stream0)
        buf77 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf75, buf71, buf72, buf69, buf73, buf74, buf76, buf10, buf77, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf79 = buf78; del buf78  # reuse
        buf80 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [conv2d_12, batch_norm_12, xout_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf79, primals_75, primals_76, primals_77, primals_78, primals_79, buf80, 131072, grid=grid(131072), stream=stream0)
        del primals_75
        buf81 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_34.run(buf81, 64, grid=grid(64), stream=stream0)
        buf82 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_35.run(buf82, 64, grid=grid(64), stream=stream0)
        buf83 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_34.run(buf83, 64, grid=grid(64), stream=stream0)
        buf84 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_35.run(buf84, 64, grid=grid(64), stream=stream0)
        buf85 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36.run(buf85, 64, grid=grid(64), stream=stream0)
        buf86 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_37.run(buf81, buf83, buf80, buf84, buf85, buf86, 524288, grid=grid(524288), stream=stream0)
        buf87 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_36.run(buf87, 64, grid=grid(64), stream=stream0)
        buf88 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf86, buf82, buf83, buf80, buf84, buf85, buf87, buf5, buf88, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf90 = buf89; del buf89  # reuse
        buf91 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_13, batch_norm_13, xout_13, hx1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39.run(buf90, primals_81, primals_82, primals_83, primals_84, primals_85, buf2, buf91, 1048576, grid=grid(1048576), stream=stream0)
        del primals_81
        buf92 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf93 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_11], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_40.run(buf91, buf92, buf93, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf92, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf95 = buf94; del buf94  # reuse
        buf96 = reinterpret_tensor(buf86, (4, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [conv2d_14, batch_norm_14, xout_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(buf95, primals_87, primals_88, primals_89, primals_90, primals_91, buf96, 524288, grid=grid(524288), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf98 = buf97; del buf97  # reuse
        buf99 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15, batch_norm_15, xout_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf98, primals_93, primals_94, primals_95, primals_96, primals_97, buf99, 131072, grid=grid(131072), stream=stream0)
        del primals_93
        del primals_97
        buf100 = buf69; del buf69  # reuse
        buf101 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_12], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_4.run(buf99, buf100, buf101, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf100, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf103 = buf102; del buf102  # reuse
        buf104 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_16, batch_norm_16, xout_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf103, primals_99, primals_100, primals_101, primals_102, primals_103, buf104, 32768, grid=grid(32768), stream=stream0)
        del primals_103
        del primals_99
        buf105 = buf58; del buf58  # reuse
        buf106 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_13], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf104, buf105, buf106, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf105, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_17, batch_norm_17, xout_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf108, primals_105, primals_106, primals_107, primals_108, primals_109, buf109, 8192, grid=grid(8192), stream=stream0)
        del primals_105
        del primals_109
        buf110 = buf47; del buf47  # reuse
        buf111 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_14], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf109, buf110, buf111, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf110, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 4, 4), (512, 16, 4, 1))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_18, batch_norm_18, xout_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf113, primals_111, primals_112, primals_113, primals_114, primals_115, buf114, 2048, grid=grid(2048), stream=stream0)
        del primals_111
        del primals_115
        buf115 = buf36; del buf36  # reuse
        buf116 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_15], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_10.run(buf114, buf115, buf116, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf115, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 32, 2, 2), (128, 4, 2, 1))
        buf118 = buf117; del buf117  # reuse
        buf119 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_19, batch_norm_19, xout_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf118, primals_117, primals_118, primals_119, primals_120, primals_121, buf119, 512, grid=grid(512), stream=stream0)
        del primals_117
        del primals_121
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_122, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 32, 2, 2), (128, 4, 2, 1))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf121, primals_123, 512, grid=grid(512), stream=stream0)
        del primals_123
        buf122 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf121, primals_124, primals_125, primals_126, primals_127, buf119, buf122, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 32, 2, 2), (128, 4, 2, 1))
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_21, batch_norm_21, xout_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf124, primals_129, primals_130, primals_131, primals_132, primals_133, buf125, 512, grid=grid(512), stream=stream0)
        del primals_129
        buf126 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_17.run(buf37, buf39, buf125, buf40, buf41, buf126, 2048, grid=grid(2048), stream=stream0)
        buf127 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf126, buf38, buf39, buf125, buf40, buf41, buf43, buf114, buf127, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 32, 4, 4), (512, 16, 4, 1))
        buf129 = buf128; del buf128  # reuse
        buf130 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [conv2d_22, batch_norm_22, xout_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf129, primals_135, primals_136, primals_137, primals_138, primals_139, buf130, 2048, grid=grid(2048), stream=stream0)
        del primals_135
        buf131 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_22.run(buf48, buf50, buf130, buf51, buf52, buf131, 8192, grid=grid(8192), stream=stream0)
        buf132 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf131, buf49, buf50, buf130, buf51, buf52, buf54, buf109, buf132, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf134 = buf133; del buf133  # reuse
        buf135 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [conv2d_23, batch_norm_23, xout_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf134, primals_141, primals_142, primals_143, primals_144, primals_145, buf135, 8192, grid=grid(8192), stream=stream0)
        del primals_141
        buf136 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_27.run(buf59, buf61, buf135, buf62, buf63, buf136, 32768, grid=grid(32768), stream=stream0)
        buf137 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf136, buf60, buf61, buf135, buf62, buf63, buf65, buf104, buf137, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf139 = buf138; del buf138  # reuse
        buf140 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [conv2d_24, batch_norm_24, xout_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf139, primals_147, primals_148, primals_149, primals_150, primals_151, buf140, 32768, grid=grid(32768), stream=stream0)
        del primals_147
        buf141 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_32.run(buf70, buf72, buf140, buf73, buf74, buf141, 131072, grid=grid(131072), stream=stream0)
        buf142 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf141, buf71, buf72, buf140, buf73, buf74, buf76, buf99, buf142, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf144 = buf143; del buf143  # reuse
        buf145 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_25, batch_norm_25, xout_25, hx2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_42.run(buf144, primals_153, primals_154, primals_155, primals_156, primals_157, buf96, buf145, 524288, grid=grid(524288), stream=stream0)
        del primals_153
        buf146 = reinterpret_tensor(buf141, (4, 128, 16, 16), (32768, 256, 16, 1), 0); del buf141  # reuse
        buf147 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_21], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_43.run(buf145, buf146, buf147, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf146, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf149 = buf148; del buf148  # reuse
        buf150 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_26, batch_norm_26, xout_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_44.run(buf149, primals_159, primals_160, primals_161, primals_162, primals_163, buf150, 262144, grid=grid(262144), stream=stream0)
        del primals_159
        del primals_163
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf152 = buf151; del buf151  # reuse
        buf153 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_27, batch_norm_27, xout_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45.run(buf152, primals_165, primals_166, primals_167, primals_168, primals_169, buf153, 65536, grid=grid(65536), stream=stream0)
        del primals_165
        del primals_169
        buf154 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf155 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_22], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_46.run(buf153, buf154, buf155, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf154, primals_170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf157 = buf156; del buf156  # reuse
        buf158 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_28, batch_norm_28, xout_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(buf157, primals_171, primals_172, primals_173, primals_174, primals_175, buf158, 16384, grid=grid(16384), stream=stream0)
        del primals_171
        del primals_175
        buf159 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf160 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_23], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_48.run(buf158, buf159, buf160, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf159, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_29, batch_norm_29, xout_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(buf162, primals_177, primals_178, primals_179, primals_180, primals_181, buf163, 4096, grid=grid(4096), stream=stream0)
        del primals_177
        del primals_181
        buf164 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        buf165 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_24], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_50.run(buf163, buf164, buf165, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf164, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 64, 2, 2), (256, 4, 2, 1))
        buf167 = buf166; del buf166  # reuse
        buf168 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_30, batch_norm_30, xout_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51.run(buf167, primals_183, primals_184, primals_185, primals_186, primals_187, buf168, 1024, grid=grid(1024), stream=stream0)
        del primals_183
        del primals_187
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_188, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 64, 2, 2), (256, 4, 2, 1))
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf170, primals_189, 1024, grid=grid(1024), stream=stream0)
        del primals_189
        buf171 = reinterpret_tensor(buf130, (4, 128, 2, 2), (512, 4, 2, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [hx_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_53.run(buf170, primals_190, primals_191, primals_192, primals_193, buf168, buf171, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 64, 2, 2), (256, 4, 2, 1))
        buf173 = buf172; del buf172  # reuse
        buf174 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_32, batch_norm_32, xout_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51.run(buf173, primals_195, primals_196, primals_197, primals_198, primals_199, buf174, 1024, grid=grid(1024), stream=stream0)
        del primals_195
        buf175 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_54.run(buf37, buf39, buf174, buf40, buf41, buf175, 4096, grid=grid(4096), stream=stream0)
        buf176 = reinterpret_tensor(buf135, (4, 128, 4, 4), (2048, 16, 4, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [hx_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_55.run(buf175, buf38, buf39, buf174, buf40, buf41, buf43, buf163, buf176, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf178 = buf177; del buf177  # reuse
        buf179 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [conv2d_33, batch_norm_33, xout_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(buf178, primals_201, primals_202, primals_203, primals_204, primals_205, buf179, 4096, grid=grid(4096), stream=stream0)
        del primals_201
        buf180 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_56.run(buf48, buf50, buf179, buf51, buf52, buf180, 16384, grid=grid(16384), stream=stream0)
        buf181 = reinterpret_tensor(buf140, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hx_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_57.run(buf180, buf49, buf50, buf179, buf51, buf52, buf54, buf158, buf181, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf183 = buf182; del buf182  # reuse
        buf184 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [conv2d_34, batch_norm_34, xout_34], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(buf183, primals_207, primals_208, primals_209, primals_210, primals_211, buf184, 16384, grid=grid(16384), stream=stream0)
        del primals_207
        buf185 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_58.run(buf59, buf61, buf184, buf62, buf63, buf185, 65536, grid=grid(65536), stream=stream0)
        buf186 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_59.run(buf185, buf60, buf61, buf184, buf62, buf63, buf65, buf153, buf186, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_35], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf188 = buf187; del buf187  # reuse
        buf189 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_35, batch_norm_35, xout_35, hx3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_60.run(buf188, primals_213, primals_214, primals_215, primals_216, primals_217, buf150, buf189, 262144, grid=grid(262144), stream=stream0)
        del primals_213
        buf190 = reinterpret_tensor(buf185, (4, 256, 8, 8), (16384, 64, 8, 1), 0); del buf185  # reuse
        buf191 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_29], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_61.run(buf189, buf190, buf191, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf190, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_36, batch_norm_36, xout_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_62.run(buf193, primals_219, primals_220, primals_221, primals_222, primals_223, buf194, 131072, grid=grid(131072), stream=stream0)
        del primals_219
        del primals_223
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf196 = buf195; del buf195  # reuse
        buf197 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_37, batch_norm_37, xout_37], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63.run(buf196, primals_225, primals_226, primals_227, primals_228, primals_229, buf197, 32768, grid=grid(32768), stream=stream0)
        del primals_225
        del primals_229
        buf198 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf199 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_30], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_64.run(buf197, buf198, buf199, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf198, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf201 = buf200; del buf200  # reuse
        buf202 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_38, batch_norm_38, xout_38], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65.run(buf201, primals_231, primals_232, primals_233, primals_234, primals_235, buf202, 8192, grid=grid(8192), stream=stream0)
        del primals_231
        del primals_235
        buf203 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        buf204 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_31], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_66.run(buf202, buf203, buf204, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf203, primals_236, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 128, 2, 2), (512, 4, 2, 1))
        buf206 = buf205; del buf205  # reuse
        buf207 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_39, batch_norm_39, xout_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67.run(buf206, primals_237, primals_238, primals_239, primals_240, primals_241, buf207, 2048, grid=grid(2048), stream=stream0)
        del primals_237
        del primals_241
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_242, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 128, 2, 2), (512, 4, 2, 1))
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_68.run(buf209, primals_243, 2048, grid=grid(2048), stream=stream0)
        del primals_243
        buf210 = reinterpret_tensor(buf179, (4, 256, 2, 2), (1024, 4, 2, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [hx_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_69.run(buf209, primals_244, primals_245, primals_246, primals_247, buf207, buf210, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 128, 2, 2), (512, 4, 2, 1))
        buf212 = buf211; del buf211  # reuse
        buf213 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_41, batch_norm_41, xout_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67.run(buf212, primals_249, primals_250, primals_251, primals_252, primals_253, buf213, 2048, grid=grid(2048), stream=stream0)
        del primals_249
        buf214 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_70.run(buf37, buf39, buf213, buf40, buf41, buf214, 8192, grid=grid(8192), stream=stream0)
        buf215 = reinterpret_tensor(buf184, (4, 256, 4, 4), (4096, 16, 4, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [hx_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_71.run(buf214, buf38, buf39, buf213, buf40, buf41, buf43, buf202, buf215, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_254, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf217 = buf216; del buf216  # reuse
        buf218 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [conv2d_42, batch_norm_42, xout_42], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65.run(buf217, primals_255, primals_256, primals_257, primals_258, primals_259, buf218, 8192, grid=grid(8192), stream=stream0)
        del primals_255
        buf219 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_72.run(buf48, buf50, buf218, buf51, buf52, buf219, 32768, grid=grid(32768), stream=stream0)
        buf220 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_73.run(buf219, buf49, buf50, buf218, buf51, buf52, buf54, buf197, buf220, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf222 = buf221; del buf221  # reuse
        buf223 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_43, batch_norm_43, xout_43, hx4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_74.run(buf222, primals_261, primals_262, primals_263, primals_264, primals_265, buf194, buf223, 131072, grid=grid(131072), stream=stream0)
        del primals_261
        buf224 = reinterpret_tensor(buf219, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf219  # reuse
        buf225 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_35], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_75.run(buf223, buf224, buf225, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf224, primals_266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf227 = buf226; del buf226  # reuse
        buf228 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_44, batch_norm_44, xout_44], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76.run(buf227, primals_267, primals_268, primals_269, primals_270, primals_271, buf228, 32768, grid=grid(32768), stream=stream0)
        del primals_267
        del primals_271
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf230 = buf229; del buf229  # reuse
        buf231 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_45, batch_norm_45, xout_45], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf230, primals_273, primals_274, primals_275, primals_276, primals_277, buf231, 16384, grid=grid(16384), stream=stream0)
        del primals_273
        del primals_277
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_278, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf233 = buf232; del buf232  # reuse
        buf234 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_46, batch_norm_46, xout_46], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf233, primals_279, primals_280, primals_281, primals_282, primals_283, buf234, 16384, grid=grid(16384), stream=stream0)
        del primals_279
        del primals_283
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_284, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf236 = buf235; del buf235  # reuse
        buf237 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_47, batch_norm_47, xout_47], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf236, primals_285, primals_286, primals_287, primals_288, primals_289, buf237, 16384, grid=grid(16384), stream=stream0)
        del primals_285
        del primals_289
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_290, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf239, primals_291, 16384, grid=grid(16384), stream=stream0)
        del primals_291
        buf240 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf239, primals_292, primals_293, primals_294, primals_295, buf237, buf240, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_296, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf242 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf242, primals_297, 16384, grid=grid(16384), stream=stream0)
        del primals_297
        buf243 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf242, primals_298, primals_299, primals_300, primals_301, buf234, buf243, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_302, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf245, primals_303, 16384, grid=grid(16384), stream=stream0)
        del primals_303
        buf246 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf245, primals_304, primals_305, primals_306, primals_307, buf231, buf246, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_308, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf248 = buf247; del buf247  # reuse
        buf249 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_51, batch_norm_51, xout_51, hx5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80.run(buf248, primals_309, primals_310, primals_311, primals_312, primals_313, buf228, buf249, 32768, grid=grid(32768), stream=stream0)
        del primals_309
        buf250 = reinterpret_tensor(buf218, (4, 512, 2, 2), (2048, 4, 2, 1), 0); del buf218  # reuse
        buf251 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_39], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_81.run(buf249, buf250, buf251, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf250, primals_314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf253 = buf252; del buf252  # reuse
        buf254 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_52, batch_norm_52, xout_52], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_82.run(buf253, primals_315, primals_316, primals_317, primals_318, primals_319, buf254, 8192, grid=grid(8192), stream=stream0)
        del primals_315
        del primals_319
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf256 = buf255; del buf255  # reuse
        buf257 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_53, batch_norm_53, xout_53], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf256, primals_321, primals_322, primals_323, primals_324, primals_325, buf257, 4096, grid=grid(4096), stream=stream0)
        del primals_321
        del primals_325
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_326, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf259 = buf258; del buf258  # reuse
        buf260 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_54, batch_norm_54, xout_54], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf259, primals_327, primals_328, primals_329, primals_330, primals_331, buf260, 4096, grid=grid(4096), stream=stream0)
        del primals_327
        del primals_331
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_332, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf262 = buf261; del buf261  # reuse
        buf263 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_55, batch_norm_55, xout_55], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf262, primals_333, primals_334, primals_335, primals_336, primals_337, buf263, 4096, grid=grid(4096), stream=stream0)
        del primals_333
        del primals_337
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_338, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf265 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_84.run(buf265, primals_339, 4096, grid=grid(4096), stream=stream0)
        del primals_339
        buf266 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_85.run(buf265, primals_340, primals_341, primals_342, primals_343, buf263, buf266, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_344, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf268 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_84.run(buf268, primals_345, 4096, grid=grid(4096), stream=stream0)
        del primals_345
        buf269 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_85.run(buf268, primals_346, primals_347, primals_348, primals_349, buf260, buf269, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_350, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 256, 2, 2), (1024, 4, 2, 1))
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_84.run(buf271, primals_351, 4096, grid=grid(4096), stream=stream0)
        del primals_351
        buf272 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_85.run(buf271, primals_352, primals_353, primals_354, primals_355, buf257, buf272, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf274 = buf273; del buf273  # reuse
        buf275 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_59, batch_norm_59, xout_59, hx6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_86.run(buf274, primals_357, primals_358, primals_359, primals_360, primals_361, buf254, buf275, 8192, grid=grid(8192), stream=stream0)
        del primals_357
        buf276 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx6up_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_87.run(buf37, buf39, buf275, buf40, buf41, buf276, 32768, grid=grid(32768), stream=stream0)
        buf277 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_43], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_88.run(buf276, buf38, buf39, buf275, buf40, buf41, buf43, buf249, buf277, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf279 = buf278; del buf278  # reuse
        buf280 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [conv2d_60, batch_norm_60, xout_60], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_76.run(buf279, primals_363, primals_364, primals_365, primals_366, primals_367, buf280, 32768, grid=grid(32768), stream=stream0)
        del primals_363
        del primals_367
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf282 = buf281; del buf281  # reuse
        buf283 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_61, batch_norm_61, xout_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf282, primals_369, primals_370, primals_371, primals_372, primals_373, buf283, 16384, grid=grid(16384), stream=stream0)
        del primals_369
        del primals_373
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_374, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf285 = buf284; del buf284  # reuse
        buf286 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_62, batch_norm_62, xout_62], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf285, primals_375, primals_376, primals_377, primals_378, primals_379, buf286, 16384, grid=grid(16384), stream=stream0)
        del primals_375
        del primals_379
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_380, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf288 = buf287; del buf287  # reuse
        buf289 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_63, batch_norm_63, xout_63], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf288, primals_381, primals_382, primals_383, primals_384, primals_385, buf289, 16384, grid=grid(16384), stream=stream0)
        del primals_381
        del primals_385
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_386, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf291, primals_387, 16384, grid=grid(16384), stream=stream0)
        del primals_387
        buf292 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf291, primals_388, primals_389, primals_390, primals_391, buf289, buf292, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_392, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf294, primals_393, 16384, grid=grid(16384), stream=stream0)
        del primals_393
        buf295 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf294, primals_394, primals_395, primals_396, primals_397, buf286, buf295, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_398, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_78.run(buf297, primals_399, 16384, grid=grid(16384), stream=stream0)
        del primals_399
        buf298 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf297, primals_400, primals_401, primals_402, primals_403, buf283, buf298, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_404, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf300 = buf299; del buf299  # reuse
        buf301 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_67, batch_norm_67, xout_67, hx5d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_80.run(buf300, primals_405, primals_406, primals_407, primals_408, primals_409, buf280, buf301, 32768, grid=grid(32768), stream=stream0)
        del primals_405
        buf302 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_89.run(buf48, buf50, buf301, buf51, buf52, buf302, 131072, grid=grid(131072), stream=stream0)
        buf303 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_90.run(buf302, buf49, buf50, buf301, buf51, buf52, buf54, buf223, buf303, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf305 = buf304; del buf304  # reuse
        buf306 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_68, batch_norm_68, xout_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_91.run(buf305, primals_411, primals_412, primals_413, primals_414, primals_415, buf306, 65536, grid=grid(65536), stream=stream0)
        del primals_411
        del primals_415
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_416, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf308 = buf307; del buf307  # reuse
        buf309 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_69, batch_norm_69, xout_69], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63.run(buf308, primals_417, primals_418, primals_419, primals_420, primals_421, buf309, 32768, grid=grid(32768), stream=stream0)
        del primals_417
        del primals_421
        buf310 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        buf311 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_48], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_64.run(buf309, buf310, buf311, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf310, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf313 = buf312; del buf312  # reuse
        buf314 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_70, batch_norm_70, xout_70], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65.run(buf313, primals_423, primals_424, primals_425, primals_426, primals_427, buf314, 8192, grid=grid(8192), stream=stream0)
        del primals_423
        del primals_427
        buf315 = buf213; del buf213  # reuse
        buf316 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_49], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_66.run(buf314, buf315, buf316, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf315, primals_428, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 128, 2, 2), (512, 4, 2, 1))
        buf318 = buf317; del buf317  # reuse
        buf319 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_71, batch_norm_71, xout_71], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67.run(buf318, primals_429, primals_430, primals_431, primals_432, primals_433, buf319, 2048, grid=grid(2048), stream=stream0)
        del primals_429
        del primals_433
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_434, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 128, 2, 2), (512, 4, 2, 1))
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_68.run(buf321, primals_435, 2048, grid=grid(2048), stream=stream0)
        del primals_435
        buf322 = empty_strided_cuda((4, 256, 2, 2), (1024, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_50], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_69.run(buf321, primals_436, primals_437, primals_438, primals_439, buf319, buf322, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_440, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (4, 128, 2, 2), (512, 4, 2, 1))
        buf324 = buf323; del buf323  # reuse
        buf325 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_73, batch_norm_73, xout_73], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67.run(buf324, primals_441, primals_442, primals_443, primals_444, primals_445, buf325, 2048, grid=grid(2048), stream=stream0)
        del primals_441
        buf326 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_4], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_70.run(buf37, buf39, buf325, buf40, buf41, buf326, 8192, grid=grid(8192), stream=stream0)
        buf327 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_71.run(buf326, buf38, buf39, buf325, buf40, buf41, buf43, buf314, buf327, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_74], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf329 = buf328; del buf328  # reuse
        buf330 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [conv2d_74, batch_norm_74, xout_74], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65.run(buf329, primals_447, primals_448, primals_449, primals_450, primals_451, buf330, 8192, grid=grid(8192), stream=stream0)
        del primals_447
        buf331 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_4], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_72.run(buf48, buf50, buf330, buf51, buf52, buf331, 32768, grid=grid(32768), stream=stream0)
        buf332 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_52], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_73.run(buf331, buf49, buf50, buf330, buf51, buf52, buf54, buf309, buf332, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf334 = buf333; del buf333  # reuse
        buf335 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_75, batch_norm_75, xout_75, hx4d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_92.run(buf334, primals_453, primals_454, primals_455, primals_456, primals_457, buf306, buf335, 65536, grid=grid(65536), stream=stream0)
        del primals_453
        buf336 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_93.run(buf59, buf61, buf335, buf62, buf63, buf336, 262144, grid=grid(262144), stream=stream0)
        buf337 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_53], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_94.run(buf336, buf60, buf61, buf335, buf62, buf63, buf65, buf189, buf337, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_458, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf339 = buf338; del buf338  # reuse
        buf340 = reinterpret_tensor(buf302, (4, 128, 16, 16), (32768, 256, 16, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [conv2d_76, batch_norm_76, xout_76], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_95.run(buf339, primals_459, primals_460, primals_461, primals_462, primals_463, buf340, 131072, grid=grid(131072), stream=stream0)
        del primals_459
        del primals_463
        # Topologically Sorted Source Nodes: [conv2d_77], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_464, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf342 = buf341; del buf341  # reuse
        buf343 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_77, batch_norm_77, xout_77], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45.run(buf342, primals_465, primals_466, primals_467, primals_468, primals_469, buf343, 65536, grid=grid(65536), stream=stream0)
        del primals_465
        del primals_469
        buf344 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf345 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_54], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_46.run(buf343, buf344, buf345, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf344, primals_470, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf347 = buf346; del buf346  # reuse
        buf348 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_78, batch_norm_78, xout_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(buf347, primals_471, primals_472, primals_473, primals_474, primals_475, buf348, 16384, grid=grid(16384), stream=stream0)
        del primals_471
        del primals_475
        buf349 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf350 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_55], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_48.run(buf348, buf349, buf350, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_79], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf349, primals_476, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf352 = buf351; del buf351  # reuse
        buf353 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_79, batch_norm_79, xout_79], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(buf352, primals_477, primals_478, primals_479, primals_480, primals_481, buf353, 4096, grid=grid(4096), stream=stream0)
        del primals_477
        del primals_481
        buf354 = buf174; del buf174  # reuse
        buf355 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_56], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_50.run(buf353, buf354, buf355, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_80], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf354, primals_482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 64, 2, 2), (256, 4, 2, 1))
        buf357 = buf356; del buf356  # reuse
        buf358 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_80, batch_norm_80, xout_80], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51.run(buf357, primals_483, primals_484, primals_485, primals_486, primals_487, buf358, 1024, grid=grid(1024), stream=stream0)
        del primals_483
        del primals_487
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_488, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 64, 2, 2), (256, 4, 2, 1))
        buf360 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf360, primals_489, 1024, grid=grid(1024), stream=stream0)
        del primals_489
        buf361 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [hx_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_53.run(buf360, primals_490, primals_491, primals_492, primals_493, buf358, buf361, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_494, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 64, 2, 2), (256, 4, 2, 1))
        buf363 = buf362; del buf362  # reuse
        buf364 = empty_strided_cuda((4, 64, 2, 2), (256, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_82, batch_norm_82, xout_82], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51.run(buf363, primals_495, primals_496, primals_497, primals_498, primals_499, buf364, 1024, grid=grid(1024), stream=stream0)
        del primals_495
        buf365 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_4], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_54.run(buf37, buf39, buf364, buf40, buf41, buf365, 4096, grid=grid(4096), stream=stream0)
        buf366 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [hx_58], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_55.run(buf365, buf38, buf39, buf364, buf40, buf41, buf43, buf353, buf366, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_83], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_500, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf368 = buf367; del buf367  # reuse
        buf369 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [conv2d_83, batch_norm_83, xout_83], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(buf368, primals_501, primals_502, primals_503, primals_504, primals_505, buf369, 4096, grid=grid(4096), stream=stream0)
        del primals_501
        buf370 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_56.run(buf48, buf50, buf369, buf51, buf52, buf370, 16384, grid=grid(16384), stream=stream0)
        buf371 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [hx_59], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_57.run(buf370, buf49, buf50, buf369, buf51, buf52, buf54, buf348, buf371, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_506, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf373 = buf372; del buf372  # reuse
        buf374 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [conv2d_84, batch_norm_84, xout_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(buf373, primals_507, primals_508, primals_509, primals_510, primals_511, buf374, 16384, grid=grid(16384), stream=stream0)
        del primals_507
        buf375 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_58.run(buf59, buf61, buf374, buf62, buf63, buf375, 65536, grid=grid(65536), stream=stream0)
        buf376 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_60], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_59.run(buf375, buf60, buf61, buf374, buf62, buf63, buf65, buf343, buf376, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_85], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf378 = buf377; del buf377  # reuse
        buf379 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_85, batch_norm_85, xout_85, hx3d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_96.run(buf378, primals_513, primals_514, primals_515, primals_516, primals_517, buf340, buf379, 131072, grid=grid(131072), stream=stream0)
        del primals_513
        buf380 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_6], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_97.run(buf70, buf72, buf379, buf73, buf74, buf380, 524288, grid=grid(524288), stream=stream0)
        buf381 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_61], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_98.run(buf380, buf71, buf72, buf379, buf73, buf74, buf76, buf145, buf381, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_86], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_518, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf383 = buf382; del buf382  # reuse
        buf384 = reinterpret_tensor(buf336, (4, 64, 32, 32), (65536, 1024, 32, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [conv2d_86, batch_norm_86, xout_86], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_99.run(buf383, primals_519, primals_520, primals_521, primals_522, primals_523, buf384, 262144, grid=grid(262144), stream=stream0)
        del primals_519
        del primals_523
        # Topologically Sorted Source Nodes: [conv2d_87], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_524, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf386 = buf385; del buf385  # reuse
        buf387 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_87, batch_norm_87, xout_87], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf386, primals_525, primals_526, primals_527, primals_528, primals_529, buf387, 131072, grid=grid(131072), stream=stream0)
        del primals_525
        del primals_529
        buf388 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf389 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_62], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_4.run(buf387, buf388, buf389, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_88], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf388, primals_530, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf391 = buf390; del buf390  # reuse
        buf392 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_88, batch_norm_88, xout_88], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf391, primals_531, primals_532, primals_533, primals_534, primals_535, buf392, 32768, grid=grid(32768), stream=stream0)
        del primals_531
        del primals_535
        buf393 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf394 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_63], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf392, buf393, buf394, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf393, primals_536, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf396 = buf395; del buf395  # reuse
        buf397 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_89, batch_norm_89, xout_89], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf396, primals_537, primals_538, primals_539, primals_540, primals_541, buf397, 8192, grid=grid(8192), stream=stream0)
        del primals_537
        del primals_541
        buf398 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        buf399 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_64], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf397, buf398, buf399, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_90], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf398, primals_542, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 32, 4, 4), (512, 16, 4, 1))
        buf401 = buf400; del buf400  # reuse
        buf402 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_90, batch_norm_90, xout_90], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf401, primals_543, primals_544, primals_545, primals_546, primals_547, buf402, 2048, grid=grid(2048), stream=stream0)
        del primals_543
        del primals_547
        buf403 = buf125; del buf125  # reuse
        buf404 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_65], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_10.run(buf402, buf403, buf404, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_91], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf403, primals_548, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (4, 32, 2, 2), (128, 4, 2, 1))
        buf406 = buf405; del buf405  # reuse
        buf407 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_91, batch_norm_91, xout_91], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf406, primals_549, primals_550, primals_551, primals_552, primals_553, buf407, 512, grid=grid(512), stream=stream0)
        del primals_549
        del primals_553
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_554, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 32, 2, 2), (128, 4, 2, 1))
        buf409 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf409, primals_555, 512, grid=grid(512), stream=stream0)
        del primals_555
        buf410 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [hx_66], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf409, primals_556, primals_557, primals_558, primals_559, buf407, buf410, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_93], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_560, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 32, 2, 2), (128, 4, 2, 1))
        buf412 = buf411; del buf411  # reuse
        buf413 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_93, batch_norm_93, xout_93], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf412, primals_561, primals_562, primals_563, primals_564, primals_565, buf413, 512, grid=grid(512), stream=stream0)
        del primals_561
        buf414 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_17.run(buf37, buf39, buf413, buf40, buf41, buf414, 2048, grid=grid(2048), stream=stream0)
        buf415 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [hx_67], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf414, buf38, buf39, buf413, buf40, buf41, buf43, buf402, buf415, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_94], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_566, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 32, 4, 4), (512, 16, 4, 1))
        buf417 = buf416; del buf416  # reuse
        buf418 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [conv2d_94, batch_norm_94, xout_94], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf417, primals_567, primals_568, primals_569, primals_570, primals_571, buf418, 2048, grid=grid(2048), stream=stream0)
        del primals_567
        buf419 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_22.run(buf48, buf50, buf418, buf51, buf52, buf419, 8192, grid=grid(8192), stream=stream0)
        buf420 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [hx_68], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf419, buf49, buf50, buf418, buf51, buf52, buf54, buf397, buf420, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_95], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, primals_572, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf422 = buf421; del buf421  # reuse
        buf423 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [conv2d_95, batch_norm_95, xout_95], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf422, primals_573, primals_574, primals_575, primals_576, primals_577, buf423, 8192, grid=grid(8192), stream=stream0)
        del primals_573
        buf424 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_7], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_27.run(buf59, buf61, buf423, buf62, buf63, buf424, 32768, grid=grid(32768), stream=stream0)
        buf425 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [hx_69], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf424, buf60, buf61, buf423, buf62, buf63, buf65, buf392, buf425, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_96], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_578, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf427 = buf426; del buf426  # reuse
        buf428 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [conv2d_96, batch_norm_96, xout_96], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf427, primals_579, primals_580, primals_581, primals_582, primals_583, buf428, 32768, grid=grid(32768), stream=stream0)
        del primals_579
        buf429 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_6], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_32.run(buf70, buf72, buf428, buf73, buf74, buf429, 131072, grid=grid(131072), stream=stream0)
        buf430 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_70], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf429, buf71, buf72, buf428, buf73, buf74, buf76, buf387, buf430, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_97], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_584, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf432 = buf431; del buf431  # reuse
        buf433 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_97, batch_norm_97, xout_97, hx2d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_100.run(buf432, primals_585, primals_586, primals_587, primals_588, primals_589, buf384, buf433, 262144, grid=grid(262144), stream=stream0)
        del primals_585
        buf434 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_7], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_101.run(buf81, buf83, buf433, buf84, buf85, buf434, 1048576, grid=grid(1048576), stream=stream0)
        buf435 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_71], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_102.run(buf434, buf82, buf83, buf433, buf84, buf85, buf87, buf91, buf435, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_98], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_590, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf437 = buf436; del buf436  # reuse
        buf438 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [conv2d_98, batch_norm_98, xout_98], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf437, primals_591, primals_592, primals_593, primals_594, primals_595, buf438, 1048576, grid=grid(1048576), stream=stream0)
        del primals_591
        del primals_595
        # Topologically Sorted Source Nodes: [conv2d_99], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_596, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf440 = buf439; del buf439  # reuse
        buf441 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_99, batch_norm_99, xout_99], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_103.run(buf440, primals_597, primals_598, primals_599, primals_600, primals_601, buf441, 262144, grid=grid(262144), stream=stream0)
        del primals_597
        del primals_601
        buf442 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf443 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_72], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_104.run(buf441, buf442, buf443, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_100], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf442, primals_602, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf445 = buf444; del buf444  # reuse
        buf446 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_100, batch_norm_100, xout_100], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105.run(buf445, primals_603, primals_604, primals_605, primals_606, primals_607, buf446, 65536, grid=grid(65536), stream=stream0)
        del primals_603
        del primals_607
        buf447 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf448 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_73], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_106.run(buf446, buf447, buf448, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_101], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf447, primals_608, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf450 = buf449; del buf449  # reuse
        buf451 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_101, batch_norm_101, xout_101], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107.run(buf450, primals_609, primals_610, primals_611, primals_612, primals_613, buf451, 16384, grid=grid(16384), stream=stream0)
        del primals_609
        del primals_613
        buf452 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        buf453 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_74], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_108.run(buf451, buf452, buf453, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_102], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf452, primals_614, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf455 = buf454; del buf454  # reuse
        buf456 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_102, batch_norm_102, xout_102], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109.run(buf455, primals_615, primals_616, primals_617, primals_618, primals_619, buf456, 4096, grid=grid(4096), stream=stream0)
        del primals_615
        del primals_619
        buf457 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        buf458 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_75], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_110.run(buf456, buf457, buf458, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_103], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf457, primals_620, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 16, 4, 4), (256, 16, 4, 1))
        buf460 = buf459; del buf459  # reuse
        buf461 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_103, batch_norm_103, xout_103], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111.run(buf460, primals_621, primals_622, primals_623, primals_624, primals_625, buf461, 1024, grid=grid(1024), stream=stream0)
        del primals_621
        del primals_625
        buf462 = empty_strided_cuda((4, 16, 2, 2), (64, 4, 2, 1), torch.float32)
        buf463 = empty_strided_cuda((4, 16, 2, 2), (64, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_76], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_112.run(buf461, buf462, buf463, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_104], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf462, primals_626, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 16, 2, 2), (64, 4, 2, 1))
        buf465 = buf464; del buf464  # reuse
        buf466 = empty_strided_cuda((4, 16, 2, 2), (64, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_104, batch_norm_104, xout_104], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113.run(buf465, primals_627, primals_628, primals_629, primals_630, primals_631, buf466, 256, grid=grid(256), stream=stream0)
        del primals_627
        del primals_631
        # Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_632, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 16, 2, 2), (64, 4, 2, 1))
        buf468 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_114.run(buf468, primals_633, 256, grid=grid(256), stream=stream0)
        del primals_633
        buf469 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [hx_77], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_115.run(buf468, primals_634, primals_635, primals_636, primals_637, buf466, buf469, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_106], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_638, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 16, 2, 2), (64, 4, 2, 1))
        buf471 = buf470; del buf470  # reuse
        buf472 = empty_strided_cuda((4, 16, 2, 2), (64, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_106, batch_norm_106, xout_106], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_113.run(buf471, primals_639, primals_640, primals_641, primals_642, primals_643, buf472, 256, grid=grid(256), stream=stream0)
        del primals_639
        buf473 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx6up_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_116.run(buf37, buf39, buf472, buf40, buf41, buf473, 1024, grid=grid(1024), stream=stream0)
        buf474 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [hx_78], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_117.run(buf473, buf38, buf39, buf472, buf40, buf41, buf43, buf461, buf474, 2048, grid=grid(2048), stream=stream0)
        del buf472
        # Topologically Sorted Source Nodes: [conv2d_107], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_644, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 16, 4, 4), (256, 16, 4, 1))
        buf476 = buf475; del buf475  # reuse
        buf477 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [conv2d_107, batch_norm_107, xout_107], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_111.run(buf476, primals_645, primals_646, primals_647, primals_648, primals_649, buf477, 1024, grid=grid(1024), stream=stream0)
        del primals_645
        buf478 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx5dup_4], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_118.run(buf48, buf50, buf477, buf51, buf52, buf478, 4096, grid=grid(4096), stream=stream0)
        buf479 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [hx_79], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_119.run(buf478, buf49, buf50, buf477, buf51, buf52, buf54, buf456, buf479, 8192, grid=grid(8192), stream=stream0)
        del buf477
        # Topologically Sorted Source Nodes: [conv2d_108], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_650, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf481 = buf480; del buf480  # reuse
        buf482 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [conv2d_108, batch_norm_108, xout_108], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_109.run(buf481, primals_651, primals_652, primals_653, primals_654, primals_655, buf482, 4096, grid=grid(4096), stream=stream0)
        del primals_651
        buf483 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx4dup_6], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_120.run(buf59, buf61, buf482, buf62, buf63, buf483, 16384, grid=grid(16384), stream=stream0)
        buf484 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [hx_80], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_121.run(buf483, buf60, buf61, buf482, buf62, buf63, buf65, buf451, buf484, 32768, grid=grid(32768), stream=stream0)
        del buf482
        # Topologically Sorted Source Nodes: [conv2d_109], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, primals_656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf486 = buf485; del buf485  # reuse
        buf487 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [conv2d_109, batch_norm_109, xout_109], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_107.run(buf486, primals_657, primals_658, primals_659, primals_660, primals_661, buf487, 16384, grid=grid(16384), stream=stream0)
        del primals_657
        buf488 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx3dup_8], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_122.run(buf70, buf72, buf487, buf73, buf74, buf488, 65536, grid=grid(65536), stream=stream0)
        buf489 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [hx_81], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_123.run(buf488, buf71, buf72, buf487, buf73, buf74, buf76, buf446, buf489, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_110], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_662, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf491 = buf490; del buf490  # reuse
        buf492 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [conv2d_110, batch_norm_110, xout_110], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_105.run(buf491, primals_663, primals_664, primals_665, primals_666, primals_667, buf492, 65536, grid=grid(65536), stream=stream0)
        del primals_663
        buf493 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup_8], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_124.run(buf81, buf83, buf492, buf84, buf85, buf493, 262144, grid=grid(262144), stream=stream0)
        buf494 = reinterpret_tensor(buf380, (4, 32, 64, 64), (131072, 4096, 64, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [hx_82], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_125.run(buf493, buf82, buf83, buf492, buf84, buf85, buf87, buf441, buf494, 524288, grid=grid(524288), stream=stream0)
        del buf492
        del buf493
        # Topologically Sorted Source Nodes: [conv2d_111], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_668, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf496 = buf495; del buf495  # reuse
        buf497 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_111, batch_norm_111, xout_111, hx1d], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_39.run(buf496, primals_669, primals_670, primals_671, primals_672, primals_673, buf438, buf497, 1048576, grid=grid(1048576), stream=stream0)
        del primals_669
        # Topologically Sorted Source Nodes: [d1], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_674, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 1, 64, 64), (4096, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [d2], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf433, primals_676, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf500 = reinterpret_tensor(buf487, (4, 1, 64, 64), (4096, 16384, 64, 1), 0); del buf487  # reuse
        buf501 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf542 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d2, d2_1, sigmoid_2], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_126.run(buf81, buf83, buf499, primals_677, buf84, buf85, buf82, buf87, buf500, buf501, buf542, 16384, grid=grid(16384), stream=stream0)
        del buf499
        del primals_677
        # Topologically Sorted Source Nodes: [d3], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf379, primals_678, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 1, 16, 16), (256, 256, 16, 1))
        buf503 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_127.run(buf503, 64, grid=grid(64), stream=stream0)
        buf504 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_128.run(buf504, 64, grid=grid(64), stream=stream0)
        buf505 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup, d3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_127.run(buf505, 64, grid=grid(64), stream=stream0)
        buf506 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_128.run(buf506, 64, grid=grid(64), stream=stream0)
        buf507 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup, d3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129.run(buf507, 64, grid=grid(64), stream=stream0)
        buf509 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d3_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_129.run(buf509, 64, grid=grid(64), stream=stream0)
        buf508 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf510 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf543 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d3, d3_1, sigmoid_3], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_130.run(buf503, buf505, buf502, primals_679, buf506, buf507, buf504, buf509, buf508, buf510, buf543, 16384, grid=grid(16384), stream=stream0)
        del buf502
        del primals_679
        # Topologically Sorted Source Nodes: [d4], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf335, primals_680, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (4, 1, 8, 8), (64, 64, 8, 1))
        buf512 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_131.run(buf512, 64, grid=grid(64), stream=stream0)
        buf513 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_132.run(buf513, 64, grid=grid(64), stream=stream0)
        buf514 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup, d4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_131.run(buf514, 64, grid=grid(64), stream=stream0)
        buf515 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_132.run(buf515, 64, grid=grid(64), stream=stream0)
        buf516 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup, d4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133.run(buf516, 64, grid=grid(64), stream=stream0)
        buf518 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d4_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_133.run(buf518, 64, grid=grid(64), stream=stream0)
        buf517 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf519 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf544 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d4, d4_1, sigmoid_4], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_134.run(buf512, buf514, buf511, primals_681, buf515, buf516, buf513, buf518, buf517, buf519, buf544, 16384, grid=grid(16384), stream=stream0)
        del buf511
        del primals_681
        # Topologically Sorted Source Nodes: [d5], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf301, primals_682, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 1, 4, 4), (16, 16, 4, 1))
        buf521 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_135.run(buf521, 64, grid=grid(64), stream=stream0)
        buf522 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_136.run(buf522, 64, grid=grid(64), stream=stream0)
        buf523 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup, d5_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_135.run(buf523, 64, grid=grid(64), stream=stream0)
        buf524 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_136.run(buf524, 64, grid=grid(64), stream=stream0)
        buf525 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [hx2dup, d5_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137.run(buf525, 64, grid=grid(64), stream=stream0)
        buf527 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d5_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_137.run(buf527, 64, grid=grid(64), stream=stream0)
        buf526 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf528 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf545 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d5, d5_1, sigmoid_5], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_138.run(buf521, buf523, buf520, primals_683, buf524, buf525, buf522, buf527, buf526, buf528, buf545, 16384, grid=grid(16384), stream=stream0)
        del primals_683
        # Topologically Sorted Source Nodes: [d6], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf275, primals_684, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (4, 1, 2, 2), (4, 4, 2, 1))
        buf530 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_139.run(buf530, 64, grid=grid(64), stream=stream0)
        buf531 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_140.run(buf531, 64, grid=grid(64), stream=stream0)
        buf532 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hx2dup, d6_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_139.run(buf532, 64, grid=grid(64), stream=stream0)
        buf533 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_140.run(buf533, 64, grid=grid(64), stream=stream0)
        buf534 = reinterpret_tensor(buf520, (64, ), (1, ), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [hx2dup, d6_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141.run(buf534, 64, grid=grid(64), stream=stream0)
        buf536 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d6_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_141.run(buf536, 64, grid=grid(64), stream=stream0)
        buf535 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf537 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf546 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d6, d6_1, sigmoid_6], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sigmoid_sub_142.run(buf530, buf532, buf529, primals_685, buf533, buf534, buf531, buf536, buf535, buf537, buf546, 16384, grid=grid(16384), stream=stream0)
        del buf529
        del primals_685
        buf538 = empty_strided_cuda((4, 6, 64, 64), (24576, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_50], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_143.run(buf498, primals_675, buf500, buf501, buf508, buf510, buf517, buf519, buf526, buf528, buf535, buf537, buf538, 98304, grid=grid(98304), stream=stream0)
        del buf500
        del buf501
        del buf508
        del buf510
        del buf517
        del buf519
        del buf526
        del buf528
        del buf535
        del buf537
        # Topologically Sorted Source Nodes: [d0], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_686, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf540 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [d0, sigmoid], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_144.run(buf540, primals_687, 16384, grid=grid(16384), stream=stream0)
        del primals_687
        buf541 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [d1, sigmoid_1], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_144.run(buf541, primals_675, 16384, grid=grid(16384), stream=stream0)
        del primals_675
    return (buf540, buf541, buf542, buf543, buf544, buf545, buf546, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_98, primals_100, primals_101, primals_102, primals_104, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_116, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_157, primals_158, primals_160, primals_161, primals_162, primals_164, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_176, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_188, primals_190, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_206, primals_208, primals_209, primals_210, primals_211, primals_212, primals_214, primals_215, primals_216, primals_217, primals_218, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_230, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_242, primals_244, primals_245, primals_246, primals_247, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_256, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_266, primals_268, primals_269, primals_270, primals_272, primals_274, primals_275, primals_276, primals_278, primals_280, primals_281, primals_282, primals_284, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_296, primals_298, primals_299, primals_300, primals_301, primals_302, primals_304, primals_305, primals_306, primals_307, primals_308, primals_310, primals_311, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_320, primals_322, primals_323, primals_324, primals_326, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_338, primals_340, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_356, primals_358, primals_359, primals_360, primals_361, primals_362, primals_364, primals_365, primals_366, primals_368, primals_370, primals_371, primals_372, primals_374, primals_376, primals_377, primals_378, primals_380, primals_382, primals_383, primals_384, primals_386, primals_388, primals_389, primals_390, primals_391, primals_392, primals_394, primals_395, primals_396, primals_397, primals_398, primals_400, primals_401, primals_402, primals_403, primals_404, primals_406, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_416, primals_418, primals_419, primals_420, primals_422, primals_424, primals_425, primals_426, primals_428, primals_430, primals_431, primals_432, primals_434, primals_436, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_446, primals_448, primals_449, primals_450, primals_451, primals_452, primals_454, primals_455, primals_456, primals_457, primals_458, primals_460, primals_461, primals_462, primals_464, primals_466, primals_467, primals_468, primals_470, primals_472, primals_473, primals_474, primals_476, primals_478, primals_479, primals_480, primals_482, primals_484, primals_485, primals_486, primals_488, primals_490, primals_491, primals_492, primals_493, primals_494, primals_496, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_506, primals_508, primals_509, primals_510, primals_511, primals_512, primals_514, primals_515, primals_516, primals_517, primals_518, primals_520, primals_521, primals_522, primals_524, primals_526, primals_527, primals_528, primals_530, primals_532, primals_533, primals_534, primals_536, primals_538, primals_539, primals_540, primals_542, primals_544, primals_545, primals_546, primals_548, primals_550, primals_551, primals_552, primals_554, primals_556, primals_557, primals_558, primals_559, primals_560, primals_562, primals_563, primals_564, primals_565, primals_566, primals_568, primals_569, primals_570, primals_571, primals_572, primals_574, primals_575, primals_576, primals_577, primals_578, primals_580, primals_581, primals_582, primals_583, primals_584, primals_586, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_596, primals_598, primals_599, primals_600, primals_602, primals_604, primals_605, primals_606, primals_608, primals_610, primals_611, primals_612, primals_614, primals_616, primals_617, primals_618, primals_620, primals_622, primals_623, primals_624, primals_626, primals_628, primals_629, primals_630, primals_632, primals_634, primals_635, primals_636, primals_637, primals_638, primals_640, primals_641, primals_642, primals_643, primals_644, primals_646, primals_647, primals_648, primals_649, primals_650, primals_652, primals_653, primals_654, primals_655, primals_656, primals_658, primals_659, primals_660, primals_661, primals_662, primals_664, primals_665, primals_666, primals_667, primals_668, primals_670, primals_671, primals_672, primals_673, primals_674, primals_676, primals_678, primals_680, primals_682, primals_684, primals_686, buf1, buf2, buf4, buf5, buf6, buf7, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf24, buf25, buf26, buf27, buf29, buf30, buf32, buf33, buf35, buf37, buf38, buf39, buf40, buf41, buf43, buf44, buf46, buf48, buf49, buf50, buf51, buf52, buf54, buf55, buf57, buf59, buf60, buf61, buf62, buf63, buf65, buf66, buf68, buf70, buf71, buf72, buf73, buf74, buf76, buf77, buf79, buf81, buf82, buf83, buf84, buf85, buf87, buf88, buf90, buf91, buf92, buf93, buf95, buf96, buf98, buf99, buf100, buf101, buf103, buf104, buf105, buf106, buf108, buf109, buf110, buf111, buf113, buf114, buf115, buf116, buf118, buf119, buf121, buf122, buf124, buf127, buf129, buf132, buf134, buf137, buf139, buf142, buf144, buf145, buf146, buf147, buf149, buf150, buf152, buf153, buf154, buf155, buf157, buf158, buf159, buf160, buf162, buf163, buf164, buf165, buf167, buf168, buf170, buf171, buf173, buf176, buf178, buf181, buf183, buf186, buf188, buf189, buf190, buf191, buf193, buf194, buf196, buf197, buf198, buf199, buf201, buf202, buf203, buf204, buf206, buf207, buf209, buf210, buf212, buf215, buf217, buf220, buf222, buf223, buf224, buf225, buf227, buf228, buf230, buf231, buf233, buf234, buf236, buf237, buf239, buf240, buf242, buf243, buf245, buf246, buf248, buf249, buf250, buf251, buf253, buf254, buf256, buf257, buf259, buf260, buf262, buf263, buf265, buf266, buf268, buf269, buf271, buf272, buf274, buf275, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf288, buf289, buf291, buf292, buf294, buf295, buf297, buf298, buf300, buf301, buf303, buf305, buf306, buf308, buf309, buf310, buf311, buf313, buf314, buf315, buf316, buf318, buf319, buf321, buf322, buf324, buf327, buf329, buf332, buf334, buf335, buf337, buf339, buf340, buf342, buf343, buf344, buf345, buf347, buf348, buf349, buf350, buf352, buf353, buf354, buf355, buf357, buf358, buf360, buf361, buf363, buf366, buf368, buf371, buf373, buf376, buf378, buf379, buf381, buf383, buf384, buf386, buf387, buf388, buf389, buf391, buf392, buf393, buf394, buf396, buf397, buf398, buf399, buf401, buf402, buf403, buf404, buf406, buf407, buf409, buf410, buf412, buf415, buf417, buf420, buf422, buf425, buf427, buf430, buf432, buf433, buf435, buf437, buf438, buf440, buf441, buf442, buf443, buf445, buf446, buf447, buf448, buf450, buf451, buf452, buf453, buf455, buf456, buf457, buf458, buf460, buf461, buf462, buf463, buf465, buf466, buf468, buf469, buf471, buf474, buf476, buf479, buf481, buf484, buf486, buf489, buf491, buf494, buf496, buf497, buf503, buf504, buf505, buf506, buf507, buf509, buf512, buf513, buf514, buf515, buf516, buf518, buf521, buf522, buf523, buf524, buf525, buf527, buf530, buf531, buf532, buf533, buf534, buf536, buf538, buf540, buf541, buf542, buf543, buf544, buf545, buf546, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((256, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((16, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((1, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((1, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((1, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((1, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((1, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((1, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
