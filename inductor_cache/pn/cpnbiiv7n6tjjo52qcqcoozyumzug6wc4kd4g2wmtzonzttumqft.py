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


# kernel path: inductor_cache/og/coge3oa5dh5wexldz4jzzubmeeoeelnc2a3y3vxbtnvxamqpxbne.py
# Topologically Sorted Source Nodes: [x, input_1, input_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_1, mul_1, mul_2, sub
#   input_2 => relu
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 48)
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


# kernel path: inductor_cache/q4/cq4a4ho3r7foswtducv4h2eed7pv7ksv3gr6j274ru7yvdgrzh2f.py
# Topologically Sorted Source Nodes: [x_1, input_5, input_6], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
#   x_1 => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_1], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 851968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 52)
    x0 = (xindex % 4096)
    x2 = xindex // 212992
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 196608*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 52, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-48) + x1) + 16384*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-48) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7z7qkmrhds5nbtvukzp7dbtr2wxtd7uewajulmmtmjipwroxfic.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_7 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 851968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 52)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/35/c35rthkkma5nbsn2rrbu2lcb6wi5iriuop6me4iqgjszgxu6xgyq.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_10 => relu_2
#   input_8 => _low_memory_max_pool2d_with_offsets
#   input_9 => add_5, mul_7, mul_8, sub_2
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_2, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_286), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 212992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x4 = xindex // 32
    x2 = ((xindex // 1024) % 52)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr1 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/xh/cxhpemzyemm3ftrnegq5kjjkrwxhcxnt524v7im722lx6xc4cdee.py
# Topologically Sorted Source Nodes: [x_2, input_13, input_14], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_13 => add_7, mul_10, mul_11, sub_3
#   input_14 => relu_3
#   x_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_3], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 56)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x3 = xindex // 57344
    x4 = (xindex % 1024)
    x5 = xindex
    tmp23 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 4096*(x2) + 212992*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 4096*(x2) + 212992*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 4096*(x2) + 212992*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 4096*(x2) + 212992*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 56, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (x4 + 1024*((-52) + x2) + 4096*x3), tmp14, other=0.0)
    tmp18 = tl.load(in_ptr2 + ((-52) + x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp13, tmp21)
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
    tl.store(out_ptr0 + (x5), tmp22, None)
    tl.store(out_ptr1 + (x5), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/hm/chmrnunkz5r2bbyobu27m4tuqbara5vg6u3f4j7yq5pr7dgxu2xc.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_15 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 56)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck5q2a4rctkicuukmjpmg2b2kfoy6othilqcbbaus5nuckwxrgke.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_16 => _low_memory_max_pool2d_with_offsets_1, getitem_3
#   input_17 => add_9, mul_13, mul_14, sub_4
#   input_18 => relu_4
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_4, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %unsqueeze_262), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x5 = xindex
    x3 = ((xindex // 256) % 56)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
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
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x5), tmp33, None)
    tl.store(out_ptr2 + (x5), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/42/c42h2jaqqserps26y7tepdeahfrcaxb5elib5asjlsb42tcse2j3.py
# Topologically Sorted Source Nodes: [x_3, input_21, input_22], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_21 => add_11, mul_16, mul_17, sub_5
#   input_22 => relu_5
#   x_3 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_2, %convolution_5], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 60)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x3 = xindex // 15360
    x4 = (xindex % 256)
    x5 = xindex
    tmp23 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x0 + 64*x1 + 1024*(x2) + 57344*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1 + 1024*(x2) + 57344*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1 + 1024*(x2) + 57344*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1 + 1024*(x2) + 57344*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 60, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (x4 + 256*((-56) + x2) + 1024*x3), tmp14, other=0.0)
    tmp18 = tl.load(in_ptr2 + ((-56) + x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp13, tmp21)
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
    tl.store(out_ptr0 + (x5), tmp22, None)
    tl.store(out_ptr1 + (x5), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crf7sk6fgjjuw5p3edzjtdplak7glu5vznf34abc4weqnnzhjzkd.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_23 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_38, %primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 60)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/sd/csdcdsbkpasmfqa5dejqmaeza3gycwnn4tmm6wuy5snpqca55lfe.py
# Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_24 => _low_memory_max_pool2d_with_offsets_2, getitem_5
#   input_25 => add_13, mul_19, mul_20, sub_6
#   input_26 => relu_6
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_4, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_4, %unsqueeze_238), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x5 = xindex
    x3 = ((xindex // 64) % 60)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
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
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
    tl.store(out_ptr2 + (x5), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/us/cus45atldt72lnzsydydmrlrirehku3q4fcdzyecsqmx7ztewrfc.py
# Topologically Sorted Source Nodes: [x_4, input_29, input_30], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_29 => add_15, mul_22, mul_23, sub_7
#   input_30 => relu_7
#   x_4 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_4, %convolution_7], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 64)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x3 = xindex // 4096
    x4 = (xindex % 64)
    x5 = xindex
    tmp23 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x0 + 32*x1 + 256*(x2) + 15360*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1 + 256*(x2) + 15360*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1 + 256*(x2) + 15360*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1 + 256*(x2) + 15360*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 64, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (x4 + 64*((-60) + x2) + 256*x3), tmp14, other=0.0)
    tmp18 = tl.load(in_ptr2 + ((-60) + x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp13, tmp21)
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
    tl.store(out_ptr0 + (x5), tmp22, None)
    tl.store(out_ptr1 + (x5), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/kg/ckg7ofswy3optmsugb42dconk7urg4dce2t2343ayh2wvgmy23d3.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_31 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_50, %primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/we/cwewrd3qudsxa3ugdxg6gmfl5jbuky6wx75c6dn5y5bc7wwmva4p.py
# Topologically Sorted Source Nodes: [input_32, input_33, input_34], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_32 => _low_memory_max_pool2d_with_offsets_3, getitem_7
#   input_33 => add_17, mul_25, mul_26, sub_8
#   input_34 => relu_8
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_8, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_6, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_6, %unsqueeze_214), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x5 = xindex
    x3 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
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
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x5), tmp33, None)
    tl.store(out_ptr2 + (x5), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/e3/ce3n5t42wknykpmronaszx576l2lh6m5dtngiqxcvwc236snad76.py
# Topologically Sorted Source Nodes: [x_5, input_37, input_38], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_37 => add_19, mul_28, mul_29, sub_9
#   input_38 => relu_9
#   x_5 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_6, %convolution_9], 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 68)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 1088
    x4 = (xindex % 16)
    x5 = xindex
    tmp23 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x0 + 16*x1 + 64*(x2) + 4096*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1 + 64*(x2) + 4096*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1 + 64*(x2) + 4096*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1 + 64*(x2) + 4096*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 68, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (x4 + 16*((-64) + x2) + 64*x3), tmp14 & xmask, other=0.0)
    tmp18 = tl.load(in_ptr2 + ((-64) + x2), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp13, tmp21)
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
    tl.store(out_ptr0 + (x5), tmp22, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z3/cz3durrx5cxjwg63fuaipttmaguj5rqycp234cevagz5ag4is6tn.py
# Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_39 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 68)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w2/cw2bbcutl3guxy2kkzgmrviw435z6p4g4mplemjivavnmbanpnbm.py
# Topologically Sorted Source Nodes: [input_40, input_41, input_42], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_40 => _low_memory_max_pool2d_with_offsets_4, getitem_9
#   input_41 => add_21, mul_31, mul_32, sub_10
#   input_42 => relu_10
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution_10, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_9 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_8, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_8, %unsqueeze_190), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x5 = xindex
    x3 = ((xindex // 4) % 68)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
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
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp33, xmask)
    tl.store(out_ptr2 + (x5), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fv/cfvnofhfpkuv4pbyugkzxydqgbjewfg3v4oofj2olysofniisbih.py
# Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_43 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_68, %primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cr/ccrhmtzubvygeyed6drfurh4bnqol4ynlhlz3fn5xwyn26majgmy.py
# Topologically Sorted Source Nodes: [x_9, input_45, input_46], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_45 => add_23, mul_34, mul_35, sub_11
#   input_46 => relu_11
#   x_9 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_4, %convolution_12], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 72)
    x0 = (xindex % 16)
    x2 = xindex // 1152
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 68, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 1088*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 72, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-68) + x1) + 64*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-68) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bl/cbljrv72piie6zyivtkfi2jrsipbowzxitvtm2ov55cljug5kqnj.py
# Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_47 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_76, %primals_77, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2yc5cwymslfwq6pl76eey7kqfw6bndjel5nfko44ars67xpg7y.py
# Topologically Sorted Source Nodes: [x_13, input_49, input_50], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_49 => add_25, mul_37, mul_38, sub_12
#   input_50 => relu_12
#   x_13 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %convolution_14], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_8, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 68)
    x0 = (xindex % 64)
    x2 = xindex // 4352
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 4096*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 68, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-64) + x1) + 256*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-64) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvuu2wpfutcelyr6vdjuumedfctdl52mu4aaqbad7hywkckqd4rv.py
# Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_51 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_84, %primals_85, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cmsibuoy4sjt7xycyieu5ybxyh2rxg7xqscugv4ns2bcue7zcfq5.py
# Topologically Sorted Source Nodes: [x_17, input_53, input_54], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_53 => add_27, mul_40, mul_41, sub_13
#   input_54 => relu_13
#   x_17 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %convolution_16], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_10, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 64)
    x0 = (xindex % 256)
    x2 = xindex // 16384
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 15360*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-60) + x1) + 1024*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-60) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdw5iifoyvq5ezlnk4w7nlr5q7rxty4aoww4udqkpatq5wtgf4s.py
# Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_55 => convolution_17
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_92, %primals_93, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/er/cer6wtbj3c6ndikdt6qscgvbdsgt6xenqzqwjayksvd4zsrty2ua.py
# Topologically Sorted Source Nodes: [x_21, input_57, input_58], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_57 => add_29, mul_43, mul_44, sub_14
#   input_58 => relu_14
#   x_21 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %convolution_18], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_12, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 60)
    x0 = (xindex % 1024)
    x2 = xindex // 61440
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 57344*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 1024*((-56) + x1) + 4096*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-56) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/dz/cdzsuvwi4aevylizn7yomhclo52z6risqeq6jlascxd4hz7frwve.py
# Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_59 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_100, %primals_101, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cngrindocs52wqoiazfjgyci5k4if4za6j4vjqwsytybhvebral5.py
# Topologically Sorted Source Nodes: [x_25, input_61, input_62], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_61 => add_31, mul_46, mul_47, sub_15
#   input_62 => relu_15
#   x_25 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %convolution_20], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_14, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 917504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 56)
    x0 = (xindex % 4096)
    x2 = xindex // 229376
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 212992*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 56, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-52) + x1) + 16384*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-52) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/nt/cntepe6r6addheab5oazkujqxpryktijiy2szdbafdyx3d4xudla.py
# Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_26 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_14, %convolution_21], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 60)
    x0 = (xindex % 4096)
    x2 = xindex // 245760
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 229376*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-56) + x1) + 16384*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-56) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/d6/cd66hixlhcp7ftzr4kzqnfgppfmdzya3baaabzscajm6w5wtmksl.py
# Topologically Sorted Source Nodes: [x_27, log_softmax], Original ATen: [aten.convolution, aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => amax, exp, sub_16, sum_1
#   x_27 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %primals_110, %primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convolution_22, [1], True), kwargs = {})
#   %sub_16 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_16,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_per_fused__log_softmax_convolution_27 = async_compile.triton('triton_per_fused__log_softmax_convolution_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_convolution_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_convolution_27(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*r2 + 49152*x1), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/b4/cb4hxyyg7mmdmox6yjtvxbsep2idvfyyrg5qlb5x72zbksdui4zp.py
# Topologically Sorted Source Nodes: [x_27, log_softmax], Original ATen: [aten.convolution, aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => log, sub_16, sub_17
#   x_27 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %primals_110, %primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_16 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %amax), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %log), kwargs = {})
triton_poi_fused__log_softmax_convolution_28 = async_compile.triton('triton_poi_fused__log_softmax_convolution_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_convolution_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_convolution_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 12)
    x0 = (xindex % 4096)
    x2 = xindex // 49152
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 4096*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + 4096*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp4 - tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/oy/coy6ohdrkabvoplzhzkwweufw7w4w4qvts4e3z5v2uxp2wx4e4gh.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_8 => getitem_1
# Graph fragment:
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_29 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 212992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111 = args
    args.clear()
    assert_size_stride(primals_1, (48, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (48, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (48, ), (1, ))
    assert_size_stride(primals_5, (48, ), (1, ))
    assert_size_stride(primals_6, (48, ), (1, ))
    assert_size_stride(primals_7, (48, ), (1, ))
    assert_size_stride(primals_8, (4, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (52, ), (1, ))
    assert_size_stride(primals_11, (52, ), (1, ))
    assert_size_stride(primals_12, (52, ), (1, ))
    assert_size_stride(primals_13, (52, ), (1, ))
    assert_size_stride(primals_14, (52, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_15, (52, ), (1, ))
    assert_size_stride(primals_16, (52, ), (1, ))
    assert_size_stride(primals_17, (52, ), (1, ))
    assert_size_stride(primals_18, (52, ), (1, ))
    assert_size_stride(primals_19, (52, ), (1, ))
    assert_size_stride(primals_20, (4, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (56, ), (1, ))
    assert_size_stride(primals_23, (56, ), (1, ))
    assert_size_stride(primals_24, (56, ), (1, ))
    assert_size_stride(primals_25, (56, ), (1, ))
    assert_size_stride(primals_26, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_27, (56, ), (1, ))
    assert_size_stride(primals_28, (56, ), (1, ))
    assert_size_stride(primals_29, (56, ), (1, ))
    assert_size_stride(primals_30, (56, ), (1, ))
    assert_size_stride(primals_31, (56, ), (1, ))
    assert_size_stride(primals_32, (4, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (60, ), (1, ))
    assert_size_stride(primals_35, (60, ), (1, ))
    assert_size_stride(primals_36, (60, ), (1, ))
    assert_size_stride(primals_37, (60, ), (1, ))
    assert_size_stride(primals_38, (60, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_39, (60, ), (1, ))
    assert_size_stride(primals_40, (60, ), (1, ))
    assert_size_stride(primals_41, (60, ), (1, ))
    assert_size_stride(primals_42, (60, ), (1, ))
    assert_size_stride(primals_43, (60, ), (1, ))
    assert_size_stride(primals_44, (4, 60, 3, 3), (540, 9, 3, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (4, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (68, ), (1, ))
    assert_size_stride(primals_59, (68, ), (1, ))
    assert_size_stride(primals_60, (68, ), (1, ))
    assert_size_stride(primals_61, (68, ), (1, ))
    assert_size_stride(primals_62, (68, 68, 1, 1), (68, 1, 1, 1))
    assert_size_stride(primals_63, (68, ), (1, ))
    assert_size_stride(primals_64, (68, ), (1, ))
    assert_size_stride(primals_65, (68, ), (1, ))
    assert_size_stride(primals_66, (68, ), (1, ))
    assert_size_stride(primals_67, (68, ), (1, ))
    assert_size_stride(primals_68, (4, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_69, (4, ), (1, ))
    assert_size_stride(primals_70, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (72, ), (1, ))
    assert_size_stride(primals_73, (72, ), (1, ))
    assert_size_stride(primals_74, (72, ), (1, ))
    assert_size_stride(primals_75, (72, ), (1, ))
    assert_size_stride(primals_76, (4, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(primals_77, (4, ), (1, ))
    assert_size_stride(primals_78, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (68, ), (1, ))
    assert_size_stride(primals_81, (68, ), (1, ))
    assert_size_stride(primals_82, (68, ), (1, ))
    assert_size_stride(primals_83, (68, ), (1, ))
    assert_size_stride(primals_84, (4, 68, 3, 3), (612, 9, 3, 1))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_87, (4, ), (1, ))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (4, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_93, (4, ), (1, ))
    assert_size_stride(primals_94, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (60, ), (1, ))
    assert_size_stride(primals_97, (60, ), (1, ))
    assert_size_stride(primals_98, (60, ), (1, ))
    assert_size_stride(primals_99, (60, ), (1, ))
    assert_size_stride(primals_100, (4, 60, 3, 3), (540, 9, 3, 1))
    assert_size_stride(primals_101, (4, ), (1, ))
    assert_size_stride(primals_102, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_103, (4, ), (1, ))
    assert_size_stride(primals_104, (56, ), (1, ))
    assert_size_stride(primals_105, (56, ), (1, ))
    assert_size_stride(primals_106, (56, ), (1, ))
    assert_size_stride(primals_107, (56, ), (1, ))
    assert_size_stride(primals_108, (4, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (12, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_111, (12, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, input_1, input_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 786432, grid=grid(786432), stream=stream0)
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf4 = empty_strided_cuda((4, 52, 64, 64), (212992, 4096, 64, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 52, 64, 64), (212992, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, input_5, input_6], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf1, buf3, primals_9, primals_10, primals_11, primals_12, primals_13, buf4, buf5, 851968, grid=grid(851968), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 52, 64, 64), (212992, 4096, 64, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf7, primals_15, 851968, grid=grid(851968), stream=stream0)
        del primals_15
        buf9 = empty_strided_cuda((4, 52, 32, 32), (53248, 1024, 32, 1), torch.float32)
        buf73 = empty_strided_cuda((4, 52, 32, 32), (53248, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_3.run(buf7, primals_16, primals_17, primals_18, primals_19, buf9, buf73, 212992, grid=grid(212992), stream=stream0)
        del primals_16
        del primals_19
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf11 = empty_strided_cuda((4, 56, 32, 32), (57344, 1024, 32, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 56, 32, 32), (57344, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, input_13, input_14], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4.run(buf7, buf10, primals_21, primals_22, primals_23, primals_24, primals_25, buf11, buf12, 229376, grid=grid(229376), stream=stream0)
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 56, 32, 32), (57344, 1024, 32, 1))
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf14, primals_27, 229376, grid=grid(229376), stream=stream0)
        del primals_27
        buf15 = empty_strided_cuda((4, 56, 16, 16), (14336, 256, 16, 1), torch.int8)
        buf16 = empty_strided_cuda((4, 56, 16, 16), (14336, 256, 16, 1), torch.float32)
        buf72 = empty_strided_cuda((4, 56, 16, 16), (14336, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_6.run(buf14, primals_28, primals_29, primals_30, primals_31, buf15, buf16, buf72, 57344, grid=grid(57344), stream=stream0)
        del primals_28
        del primals_31
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf18 = empty_strided_cuda((4, 60, 16, 16), (15360, 256, 16, 1), torch.float32)
        buf19 = empty_strided_cuda((4, 60, 16, 16), (15360, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, input_21, input_22], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7.run(buf14, buf17, primals_33, primals_34, primals_35, primals_36, primals_37, buf18, buf19, 61440, grid=grid(61440), stream=stream0)
        del primals_33
        del primals_37
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 60, 16, 16), (15360, 256, 16, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_8.run(buf21, primals_39, 61440, grid=grid(61440), stream=stream0)
        del primals_39
        buf22 = empty_strided_cuda((4, 60, 8, 8), (3840, 64, 8, 1), torch.int8)
        buf23 = empty_strided_cuda((4, 60, 8, 8), (3840, 64, 8, 1), torch.float32)
        buf71 = empty_strided_cuda((4, 60, 8, 8), (3840, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_9.run(buf21, primals_40, primals_41, primals_42, primals_43, buf22, buf23, buf71, 15360, grid=grid(15360), stream=stream0)
        del primals_40
        del primals_43
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 8, 8), (256, 64, 8, 1))
        buf25 = reinterpret_tensor(buf10, (4, 64, 8, 8), (4096, 64, 8, 1), 0); del buf10  # reuse
        buf26 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, input_29, input_30], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10.run(buf21, buf24, primals_45, primals_46, primals_47, primals_48, primals_49, buf25, buf26, 16384, grid=grid(16384), stream=stream0)
        del buf24
        del primals_45
        del primals_49
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_11.run(buf28, primals_51, 16384, grid=grid(16384), stream=stream0)
        del primals_51
        buf29 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.int8)
        buf30 = reinterpret_tensor(buf17, (4, 64, 4, 4), (1024, 16, 4, 1), 0); del buf17  # reuse
        buf70 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33, input_34], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_12.run(buf28, primals_52, primals_53, primals_54, primals_55, buf29, buf30, buf70, 4096, grid=grid(4096), stream=stream0)
        del primals_52
        del primals_55
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 4, 4, 4), (64, 16, 4, 1))
        buf32 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 68, 4, 4), (1088, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, input_37, input_38], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13.run(buf28, buf31, primals_57, primals_58, primals_59, primals_60, primals_61, buf32, buf33, 4352, grid=grid(4352), stream=stream0)
        del buf31
        del primals_57
        del primals_61
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 68, 4, 4), (1088, 16, 4, 1))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf35, primals_63, 4352, grid=grid(4352), stream=stream0)
        del primals_63
        buf36 = empty_strided_cuda((4, 68, 2, 2), (272, 4, 2, 1), torch.int8)
        buf37 = empty_strided_cuda((4, 68, 2, 2), (272, 4, 2, 1), torch.float32)
        buf69 = empty_strided_cuda((4, 68, 2, 2), (272, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40, input_41, input_42], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_15.run(buf35, primals_64, primals_65, primals_66, primals_67, buf36, buf37, buf69, 1088, grid=grid(1088), stream=stream0)
        del primals_64
        del primals_67
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 4, 2, 2), (16, 4, 2, 1))
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(buf39, primals_69, 64, grid=grid(64), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = empty_strided_cuda((4, 72, 4, 4), (1152, 16, 4, 1), torch.float32)
        buf42 = empty_strided_cuda((4, 72, 4, 4), (1152, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, input_45, input_46], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17.run(buf32, buf40, primals_71, primals_72, primals_73, primals_74, primals_75, buf41, buf42, 4608, grid=grid(4608), stream=stream0)
        del buf40
        del primals_71
        del primals_75
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 4, 4, 4), (64, 16, 4, 1))
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_18.run(buf44, primals_77, 256, grid=grid(256), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_78, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf45, (4, 4, 8, 8), (256, 64, 8, 1))
        buf46 = empty_strided_cuda((4, 68, 8, 8), (4352, 64, 8, 1), torch.float32)
        buf47 = empty_strided_cuda((4, 68, 8, 8), (4352, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, input_49, input_50], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf25, buf45, primals_79, primals_80, primals_81, primals_82, primals_83, buf46, buf47, 17408, grid=grid(17408), stream=stream0)
        del buf45
        del primals_79
        del primals_83
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 4, 8, 8), (256, 64, 8, 1))
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf49, primals_85, 1024, grid=grid(1024), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_86, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf51 = reinterpret_tensor(buf3, (4, 64, 16, 16), (16384, 256, 16, 1), 0); del buf3  # reuse
        buf52 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_17, input_53, input_54], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21.run(buf18, buf50, primals_87, primals_88, primals_89, primals_90, primals_91, buf51, buf52, 65536, grid=grid(65536), stream=stream0)
        del buf50
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf54, primals_93, 4096, grid=grid(4096), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf55, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf56 = empty_strided_cuda((4, 60, 32, 32), (61440, 1024, 32, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 60, 32, 32), (61440, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_21, input_57, input_58], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23.run(buf11, buf55, primals_95, primals_96, primals_97, primals_98, primals_99, buf56, buf57, 245760, grid=grid(245760), stream=stream0)
        del primals_95
        del primals_99
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf59, primals_101, 16384, grid=grid(16384), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_102, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf60, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf61 = empty_strided_cuda((4, 56, 64, 64), (229376, 4096, 64, 1), torch.float32)
        buf62 = empty_strided_cuda((4, 56, 64, 64), (229376, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, input_61, input_62], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25.run(buf4, buf60, primals_103, primals_104, primals_105, primals_106, primals_107, buf61, buf62, 917504, grid=grid(917504), stream=stream0)
        del buf60
        del primals_103
        del primals_107
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf64 = empty_strided_cuda((4, 60, 64, 64), (245760, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf61, buf63, primals_109, buf64, 983040, grid=grid(983040), stream=stream0)
        del buf63
        del primals_109
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf66 = reinterpret_tensor(buf55, (4, 1, 64, 64), (4096, 16384, 64, 1), 0); del buf55  # reuse
        buf67 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, log_softmax], Original ATen: [aten.convolution, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_convolution_27.run(buf65, primals_111, buf66, buf67, 16384, 12, grid=grid(16384), stream=stream0)
        buf68 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_27, log_softmax], Original ATen: [aten.convolution, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_convolution_28.run(buf68, primals_111, buf66, buf67, 196608, grid=grid(196608), stream=stream0)
        del buf66
        del buf67
        del primals_111
        buf8 = empty_strided_cuda((4, 52, 32, 32), (53248, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_29.run(buf7, buf8, 212992, grid=grid(212992), stream=stream0)
    return (buf68, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, primals_53, primals_54, primals_56, primals_58, primals_59, primals_60, primals_62, primals_65, primals_66, primals_68, primals_70, primals_72, primals_73, primals_74, primals_76, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_96, primals_97, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_108, primals_110, buf1, buf2, buf4, buf5, buf7, buf8, buf9, buf11, buf12, buf14, buf15, buf16, buf18, buf19, buf21, buf22, buf23, buf25, buf26, buf28, buf29, buf30, buf32, buf33, buf35, buf36, buf37, buf39, buf41, buf42, buf44, buf46, buf47, buf49, buf51, buf52, buf54, buf56, buf57, buf59, buf61, buf62, buf64, buf68, buf69, buf70, buf71, buf72, buf73, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((48, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((52, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((60, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 60, 3, 3), (540, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((68, 68, 1, 1), (68, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, 68, 3, 3), (612, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((4, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((4, 60, 3, 3), (540, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((12, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
