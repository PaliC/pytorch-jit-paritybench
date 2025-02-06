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


# kernel path: inductor_cache/4x/c4xqapr7wxnipc6zjb6qe6t3acqhbzkcmc6hpdxgbg3wn2q7luox.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: inductor_cache/j6/cj6t2ujxdzx2f47cxu4oqjdyrlpoewwfcosnkg36kj5rpgahchzb.py
# Topologically Sorted Source Nodes: [e2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   e2 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6g/c6gxeg3hrg6zzcue64wx553rqdrfb3zkjk5dysaqhl2anxnswmod.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => add_5, mul_7, mul_8, sub_2
#   input_9 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3n/c3nh55p53bm75z7lksgmeptupjumup6zpexv377rx3xsh2k4midl.py
# Topologically Sorted Source Nodes: [e3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   e3 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ef/cefwp2j7yb2qt65kicwvvsocmtqva7v46cl2soloygnmkbmw6rik.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_13 => convolution_4
#   input_14 => add_9, mul_13, mul_14, sub_4
#   input_15 => relu_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/zq/czqax677ca5u7pxo5deqpqa5cq2mmsmw7ghqji6scpatesgfufoj.py
# Topologically Sorted Source Nodes: [e4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   e4 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_5 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pr/cprvpsvaw6bvizqpgkzorws3ckua7zgfoob3jnonkmv2bz3d6eje.py
# Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => convolution_6
#   input_20 => add_13, mul_19, mul_20, sub_6
#   input_21 => relu_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_38, %primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/no/cno5ehngtryjiugkls23a2phn56zfpwzx5i7zvbgfb2d6gul3ltr.py
# Topologically Sorted Source Nodes: [e5], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   e5 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3w/c3wp3rps5a3ic43hjr6fyc5mqpjn2njqhg3m4g6bhny3ak5eyhvb.py
# Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_25 => convolution_8
#   input_26 => add_17, mul_25, mul_26, sub_8
#   input_27 => relu_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_50, %primals_51, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 1024)
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


# kernel path: inductor_cache/y3/cy3h6lm5wpoymeh34nday6c36f5divt5u7whemb6dbv7xrqfoikc.py
# Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_28 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_56, %primals_57, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/5v/c5vfti4ij27w2moe3b2i7elwadubbx5feaqu6lsjvcrvqcu3bsqb.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_31 => add_20, add_21, convert_element_type_20, convert_element_type_21, iota, mul_30, mul_31
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, 0), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.float32), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_20, 0.0), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.5), kwargs = {})
#   %convert_element_type_21 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_31, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_10 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gi/cgidnrmrrqj6qqdgystvepmmgeu3oj6g2g4jbdrrbdqhf3nsluwf.py
# Topologically Sorted Source Nodes: [input_29, input_30, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_29 => add_19, mul_28, mul_29, sub_9
#   input_30 => relu_9
#   input_31 => _unsafe_index
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %unsqueeze_80, %convert_element_type_21]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex // 64
    x2 = ((xindex // 64) % 1024)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 4*tmp4 + 16*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/is/cisgychasy5ryrynbjml3jcl6ewgmxzdxqe5aeqvr5bxyvt2wcao.py
# Topologically Sorted Source Nodes: [input_35, input_36, input_37, input_38, add, psi], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add => add_30
#   input_35 => convolution_11
#   input_36 => add_27, mul_38, mul_39, sub_11
#   input_37 => convolution_12
#   input_38 => add_29, mul_41, mul_42, sub_12
#   psi => relu_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_68, %primals_69, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_90), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_92), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %unsqueeze_94), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %unsqueeze_96), kwargs = {})
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_74, %primals_75, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_98), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_100), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_102), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %unsqueeze_104), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %add_29), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp5, None)
    tl.store(in_out_ptr2 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/72/c72j7x5emxbr7aykh7atopag36ms3q6s6dev7jueziqk23r27hzp.py
# Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
# Source node to ATen node mapping:
#   input_39 => convolution_13
#   input_40 => add_32, mul_44, mul_45, sub_13
#   input_41 => sigmoid
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_80, %primals_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_106), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_108), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_110), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_112), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_32,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp19 = tmp16 * tmp18
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csma53kucapkhacozgvac6mfojrorigwbg7gestbkdjdhfaljear.py
# Topologically Sorted Source Nodes: [d5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   d5 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_46, %relu_10], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 1024)
    x0 = (xindex % 64)
    x2 = xindex // 65536
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 32768*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1024, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 64*((-512) + x1) + 32768*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqb665gchkyswl47mublewf4wboqhjsbrnt3o6jlxffkdhmy3k33.py
# Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_45 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_92, %primals_93, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/iu/ciuz2kygh6h3fcs7xifyl4hxzqola7cpbeshizeh3panui3nyi7t.py
# Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_48 => add_37, add_38, convert_element_type_36, convert_element_type_37, iota_2, mul_53, mul_54
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, 0), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_37, torch.float32), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_36, 0.0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, 0.5), kwargs = {})
#   %convert_element_type_37 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_54, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_16 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/dl/cdljs26ppaoi5tv5zu3ugetamls4zmeepqkmkgq6ovc3y4jxyjc2.py
# Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_46 => add_36, mul_51, mul_52, sub_15
#   input_47 => relu_13
#   input_48 => _unsafe_index_1
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_122), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_124), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_126), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_128), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_13, [None, None, %unsqueeze_129, %convert_element_type_37]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex // 256
    x2 = ((xindex // 256) % 512)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4 + 64*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/n5/cn5y7mlmuuxi6vzxm7rruupa2zmrpzw5da3eq4tqaj22fzfidfa6.py
# Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, add_1, psi_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_47
#   input_52 => convolution_17
#   input_53 => add_44, mul_61, mul_62, sub_17
#   input_54 => convolution_18
#   input_55 => add_46, mul_64, mul_65, sub_18
#   psi_1 => relu_15
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_104, %primals_105, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_139), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_141), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_143), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_145), kwargs = {})
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_110, %primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_147), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_149), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_151), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_153), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %add_46), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp5, None)
    tl.store(in_out_ptr2 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/w7/cw7nqet7djxjhiooaftpapttsrdpxyrpvlyyyktdkqadnw7uvssy.py
# Topologically Sorted Source Nodes: [input_56, input_57, input_58], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
# Source node to ATen node mapping:
#   input_56 => convolution_19
#   input_57 => add_49, mul_67, mul_68, sub_19
#   input_58 => sigmoid_1
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_116, %primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_155), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_157), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_159), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_161), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_49,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp19 = tmp16 * tmp18
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/25/c2576fu4zkfrj6r6zeee2ygq2xpuysxhb4arqgyzalwhoipoqbj4.py
# Topologically Sorted Source Nodes: [d4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   d4 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_69, %relu_14], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 512)
    x0 = (xindex % 256)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 65536*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 512, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 256*((-256) + x1) + 65536*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/e4/ce4ooqvaglcr62bnfm3nashzxsixtvbcsfifsqlm7f55jwrji7vg.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_62 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %primals_128, %primals_129, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuq5yn6mjcqbwwfm4pni5mfeleb4ldyfmz6c3xtupanfrjpi6lnu.py
# Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_65 => add_54, add_55, convert_element_type_52, convert_element_type_53, iota_4, mul_76, mul_77
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, 0), kwargs = {})
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_54, torch.float32), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_52, 0.0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, 0.5), kwargs = {})
#   %convert_element_type_53 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_77, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_22 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: inductor_cache/7i/c7ioxjruhnci73c2fh7ilqliadawgvvr3klkymezfuvb4vlngtkv.py
# Topologically Sorted Source Nodes: [input_63, input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_63 => add_53, mul_74, mul_75, sub_21
#   input_64 => relu_17
#   input_65 => _unsafe_index_2
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_171), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_173), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %unsqueeze_175), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %unsqueeze_177), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_17, [None, None, %unsqueeze_178, %convert_element_type_53]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex // 1024
    x2 = ((xindex // 1024) % 256)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/qh/cqhnadlocuisfhstmzxb63puhclya2wvg4erfapaukxnnfenpo3k.py
# Topologically Sorted Source Nodes: [input_69, input_70, input_71, input_72, add_2, psi_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_2 => add_64
#   input_69 => convolution_23
#   input_70 => add_61, mul_84, mul_85, sub_23
#   input_71 => convolution_24
#   input_72 => add_63, mul_87, mul_88, sub_24
#   psi_2 => relu_19
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_140, %primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_188), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_190), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %unsqueeze_192), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %unsqueeze_194), kwargs = {})
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_146, %primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_196), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_198), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_200), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_202), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %add_63), kwargs = {})
#   %relu_19 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_64,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp5, None)
    tl.store(in_out_ptr2 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/j6/cj6jdmj446hcp45f5bq6pe52lhfmhsxjz2yukfggf7rth753z7nf.py
# Topologically Sorted Source Nodes: [input_73, input_74, input_75], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
# Source node to ATen node mapping:
#   input_73 => convolution_25
#   input_74 => add_66, mul_90, mul_91, sub_25
#   input_75 => sigmoid_2
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %primals_152, %primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_204), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_206), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %unsqueeze_208), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %unsqueeze_210), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_66,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp19 = tmp16 * tmp18
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr0 + (x0), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ok/cokgd4h57gpmsthsfyohsewpslyoh7e2n7huuhqvhsx2rrjhma66.py
# Topologically Sorted Source Nodes: [d3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   d3 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_92, %relu_18], 1), kwargs = {})
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 256)
    x0 = (xindex % 1024)
    x2 = xindex // 262144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 131072*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 1024*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 1024*((-128) + x1) + 131072*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckxbxzeld6hm5yl5ty2ksatr3urihp4tw3i4nd64c53xuxtlgn35.py
# Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_79 => convolution_27
# Graph fragment:
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_164, %primals_165, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/p2/cp2axlddyodannmxhetez2i6b3jovc7dyfdovrb3dvux6eo6svcy.py
# Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_82 => add_71, add_72, convert_element_type_68, convert_element_type_69, iota_6, mul_100, mul_99
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, 0), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, 0.0), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, 0.5), kwargs = {})
#   %convert_element_type_69 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_28 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_28(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: inductor_cache/yg/cygyy6kpawrxjxp6hwmfedgijg4mjztgsvny3mobfi7grk25pdej.py
# Topologically Sorted Source Nodes: [input_80, input_81, input_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_80 => add_70, mul_97, mul_98, sub_27
#   input_81 => relu_21
#   input_82 => _unsafe_index_3
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_220), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_222), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_224), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_226), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_21, [None, None, %unsqueeze_227, %convert_element_type_69]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex // 4096
    x2 = ((xindex // 4096) % 128)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikda4wp7blckzmcyvvv7sjp5jeufy7liqxvnlg52fbtadjg7pgr.py
# Topologically Sorted Source Nodes: [input_86, input_87, input_88, input_89, add_3, psi_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_3 => add_81
#   input_86 => convolution_29
#   input_87 => add_78, mul_107, mul_108, sub_29
#   input_88 => convolution_30
#   input_89 => add_80, mul_110, mul_111, sub_30
#   psi_3 => relu_23
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %primals_176, %primals_177, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_237), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_239), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %unsqueeze_241), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %unsqueeze_243), kwargs = {})
#   %convolution_30 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_182, %primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_245), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_247), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %unsqueeze_249), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %unsqueeze_251), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %add_80), kwargs = {})
#   %relu_23 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_81,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp5, None)
    tl.store(in_out_ptr2 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/5y/c5yhalewamem7lqhjnnjwijzblsqkxpkqbwdxmsqfghqohyegn2j.py
# Topologically Sorted Source Nodes: [input_90, input_91, input_92], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
# Source node to ATen node mapping:
#   input_90 => convolution_31
#   input_91 => add_83, mul_113, mul_114, sub_31
#   input_92 => sigmoid_3
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %primals_188, %primals_189, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_253), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_255), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_257), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_259), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_83,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp19 = tmp16 * tmp18
    tmp22 = tmp19 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr0 + (x0), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/kj/ckj3yrxadmge3pdq5vycnzccbcmmewcpqwa4v75nfq4r3jg6lzmx.py
# Topologically Sorted Source Nodes: [d2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   d2 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_115, %relu_22], 1), kwargs = {})
triton_poi_fused_cat_32 = async_compile.triton('triton_poi_fused_cat_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 128)
    x0 = (xindex % 4096)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 4096*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 4096*((-64) + x1) + 262144*x2), tmp10, other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/jh/cjheydrtepqw2acguuy7eah3uzoe73vy7n7nbqhj7myrct4g6kqu.py
# Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_4 => convolution_34
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %primals_206, %primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_33 = async_compile.triton('triton_poi_fused_convolution_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_33(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, ), (1, ))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_57, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, ), (1, ))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, ), (1, ))
    assert_size_stride(primals_62, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_68, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (1, ), (1, ))
    assert_size_stride(primals_82, (1, ), (1, ))
    assert_size_stride(primals_83, (1, ), (1, ))
    assert_size_stride(primals_84, (1, ), (1, ))
    assert_size_stride(primals_85, (1, ), (1, ))
    assert_size_stride(primals_86, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_117, (1, ), (1, ))
    assert_size_stride(primals_118, (1, ), (1, ))
    assert_size_stride(primals_119, (1, ), (1, ))
    assert_size_stride(primals_120, (1, ), (1, ))
    assert_size_stride(primals_121, (1, ), (1, ))
    assert_size_stride(primals_122, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, ), (1, ))
    assert_size_stride(primals_128, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (256, ), (1, ))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (64, ), (1, ))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_147, (64, ), (1, ))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_153, (1, ), (1, ))
    assert_size_stride(primals_154, (1, ), (1, ))
    assert_size_stride(primals_155, (1, ), (1, ))
    assert_size_stride(primals_156, (1, ), (1, ))
    assert_size_stride(primals_157, (1, ), (1, ))
    assert_size_stride(primals_158, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_177, (32, ), (1, ))
    assert_size_stride(primals_178, (32, ), (1, ))
    assert_size_stride(primals_179, (32, ), (1, ))
    assert_size_stride(primals_180, (32, ), (1, ))
    assert_size_stride(primals_181, (32, ), (1, ))
    assert_size_stride(primals_182, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_183, (32, ), (1, ))
    assert_size_stride(primals_184, (32, ), (1, ))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, ), (1, ))
    assert_size_stride(primals_188, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_189, (1, ), (1, ))
    assert_size_stride(primals_190, (1, ), (1, ))
    assert_size_stride(primals_191, (1, ), (1, ))
    assert_size_stride(primals_192, (1, ), (1, ))
    assert_size_stride(primals_193, (1, ), (1, ))
    assert_size_stride(primals_194, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_201, (64, ), (1, ))
    assert_size_stride(primals_202, (64, ), (1, ))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_207, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 1048576, grid=grid(1048576), stream=stream0)
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 1048576, grid=grid(1048576), stream=stream0)
        del primals_13
        del primals_9
        buf6 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [e2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf5, buf6, buf7, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, primals_15, primals_16, primals_17, primals_18, primals_19, buf10, 524288, grid=grid(524288), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf12, primals_21, primals_22, primals_23, primals_24, primals_25, buf13, 524288, grid=grid(524288), stream=stream0)
        del primals_21
        del primals_25
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [e3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf13, buf14, buf15, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf14, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf17, primals_27, primals_28, primals_29, primals_30, primals_31, buf18, 262144, grid=grid(262144), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf20, primals_33, primals_34, primals_35, primals_36, primals_37, buf21, 262144, grid=grid(262144), stream=stream0)
        del primals_33
        del primals_37
        buf22 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf23 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [e4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_5.run(buf21, buf22, buf23, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf22, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf25, primals_39, primals_40, primals_41, primals_42, primals_43, buf26, 131072, grid=grid(131072), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf28, primals_45, primals_46, primals_47, primals_48, primals_49, buf29, 131072, grid=grid(131072), stream=stream0)
        del primals_45
        del primals_49
        buf30 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [e5], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf29, buf30, buf31, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf30, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf33, primals_51, primals_52, primals_53, primals_54, primals_55, buf34, 65536, grid=grid(65536), stream=stream0)
        del primals_51
        del primals_55
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf36, primals_57, 65536, grid=grid(65536), stream=stream0)
        del primals_57
        buf37 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_10.run(buf37, 8, grid=grid(8), stream=stream0)
        buf38 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_11.run(buf37, buf36, primals_58, primals_59, primals_60, primals_61, buf38, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf40 = buf39; del buf39  # reuse
        buf41 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33, input_34], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf40, primals_63, primals_64, primals_65, primals_66, primals_67, buf41, 131072, grid=grid(131072), stream=stream0)
        del primals_63
        del primals_67
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 256, 8, 8), (16384, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf29, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf43 = buf42; del buf42  # reuse
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36, input_37, input_38, add, psi], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12.run(buf43, buf45, buf47, primals_69, primals_75, primals_70, primals_71, primals_72, primals_73, primals_76, primals_77, primals_78, primals_79, 65536, grid=grid(65536), stream=stream0)
        del primals_69
        del primals_73
        del primals_75
        del primals_79
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 1, 8, 8), (64, 64, 8, 1))
        buf49 = buf48; del buf48  # reuse
        buf50 = empty_strided_cuda((4, 1, 8, 8), (64, 256, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_13.run(buf49, primals_81, primals_82, primals_83, primals_84, primals_85, buf50, 256, grid=grid(256), stream=stream0)
        del primals_81
        buf51 = empty_strided_cuda((4, 1024, 8, 8), (65536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf29, buf50, buf41, buf51, 262144, grid=grid(262144), stream=stream0)
        del buf50
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf53, primals_87, primals_88, primals_89, primals_90, primals_91, buf54, 131072, grid=grid(131072), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf56, primals_93, 131072, grid=grid(131072), stream=stream0)
        del primals_93
        buf57 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_16.run(buf57, 16, grid=grid(16), stream=stream0)
        buf58 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_17.run(buf57, buf56, primals_94, primals_95, primals_96, primals_97, buf58, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf60 = buf59; del buf59  # reuse
        buf61 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, input_50, input_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf60, primals_99, primals_100, primals_101, primals_102, primals_103, buf61, 262144, grid=grid(262144), stream=stream0)
        del primals_103
        del primals_99
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 16, 16), (32768, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf21, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf63 = buf62; del buf62  # reuse
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, add_1, psi_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_18.run(buf63, buf65, buf67, primals_105, primals_111, primals_106, primals_107, primals_108, primals_109, primals_112, primals_113, primals_114, primals_115, 131072, grid=grid(131072), stream=stream0)
        del primals_105
        del primals_109
        del primals_111
        del primals_115
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1, 16, 16), (256, 256, 16, 1))
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((4, 1, 16, 16), (256, 1024, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57, input_58], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_19.run(buf69, primals_117, primals_118, primals_119, primals_120, primals_121, buf70, 1024, grid=grid(1024), stream=stream0)
        del primals_117
        buf71 = empty_strided_cuda((4, 512, 16, 16), (131072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf21, buf70, buf61, buf71, 524288, grid=grid(524288), stream=stream0)
        del buf70
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf73 = buf72; del buf72  # reuse
        buf74 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60, input_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf73, primals_123, primals_124, primals_125, primals_126, primals_127, buf74, 262144, grid=grid(262144), stream=stream0)
        del primals_123
        del primals_127
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(buf76, primals_129, 262144, grid=grid(262144), stream=stream0)
        del primals_129
        buf77 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_22.run(buf77, 32, grid=grid(32), stream=stream0)
        buf78 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_63, input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_23.run(buf77, buf76, primals_130, primals_131, primals_132, primals_133, buf78, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf80, primals_135, primals_136, primals_137, primals_138, primals_139, buf81, 524288, grid=grid(524288), stream=stream0)
        del primals_135
        del primals_139
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 64, 32, 32), (65536, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf13, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf83 = buf82; del buf82  # reuse
        buf85 = buf84; del buf84  # reuse
        buf86 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_69, input_70, input_71, input_72, add_2, psi_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf83, buf85, buf87, primals_141, primals_147, primals_142, primals_143, primals_144, primals_145, primals_148, primals_149, primals_150, primals_151, 262144, grid=grid(262144), stream=stream0)
        del primals_141
        del primals_145
        del primals_147
        del primals_151
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty_strided_cuda((4, 1, 32, 32), (1024, 4096, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_73, input_74, input_75], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_25.run(buf89, primals_153, primals_154, primals_155, primals_156, primals_157, buf90, 4096, grid=grid(4096), stream=stream0)
        del primals_153
        buf91 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf13, buf90, buf81, buf91, 1048576, grid=grid(1048576), stream=stream0)
        del buf90
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, input_77, input_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf93, primals_159, primals_160, primals_161, primals_162, primals_163, buf94, 524288, grid=grid(524288), stream=stream0)
        del primals_159
        del primals_163
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf96, primals_165, 524288, grid=grid(524288), stream=stream0)
        del primals_165
        buf97 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_28.run(buf97, 64, grid=grid(64), stream=stream0)
        buf98 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81, input_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_relu_29.run(buf97, buf96, primals_166, primals_167, primals_168, primals_169, buf98, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf100 = buf99; del buf99  # reuse
        buf101 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, input_84, input_85], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf100, primals_171, primals_172, primals_173, primals_174, primals_175, buf101, 1048576, grid=grid(1048576), stream=stream0)
        del primals_171
        del primals_175
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 32, 64, 64), (131072, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf5, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf103 = buf102; del buf102  # reuse
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [input_86, input_87, input_88, input_89, add_3, psi_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_30.run(buf103, buf105, buf107, primals_177, primals_183, primals_178, primals_179, primals_180, primals_181, primals_184, primals_185, primals_186, primals_187, 524288, grid=grid(524288), stream=stream0)
        del primals_177
        del primals_181
        del primals_183
        del primals_187
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf109 = buf108; del buf108  # reuse
        buf110 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_90, input_91, input_92], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_31.run(buf109, primals_189, primals_190, primals_191, primals_192, primals_193, buf110, 16384, grid=grid(16384), stream=stream0)
        del primals_189
        buf111 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [d2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_32.run(buf5, buf110, buf101, buf111, 2097152, grid=grid(2097152), stream=stream0)
        del buf110
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_93, input_94, input_95], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf113, primals_195, primals_196, primals_197, primals_198, primals_199, buf114, 1048576, grid=grid(1048576), stream=stream0)
        del primals_195
        del primals_199
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97, input_98], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf116, primals_201, primals_202, primals_203, primals_204, primals_205, buf117, 1048576, grid=grid(1048576), stream=stream0)
        del primals_201
        del primals_205
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_33.run(buf119, primals_207, 16384, grid=grid(16384), stream=stream0)
        del primals_207
    return (buf119, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_70, primals_71, primals_72, primals_74, primals_76, primals_77, primals_78, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_116, primals_118, primals_119, primals_120, primals_121, primals_122, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_140, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_152, primals_154, primals_155, primals_156, primals_157, primals_158, primals_160, primals_161, primals_162, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_176, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_188, primals_190, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_200, primals_202, primals_203, primals_204, primals_206, buf1, buf2, buf4, buf5, buf6, buf7, buf9, buf10, buf12, buf13, buf14, buf15, buf17, buf18, buf20, buf21, buf22, buf23, buf25, buf26, buf28, buf29, buf30, buf31, buf33, buf34, buf36, buf37, buf38, buf40, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf54, buf56, buf57, buf58, buf60, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf74, buf76, buf77, buf78, buf80, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf94, buf96, buf97, buf98, buf100, buf101, buf103, buf105, buf107, buf109, buf111, buf113, buf114, buf116, buf117, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
