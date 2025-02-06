# AOT ID: ['14_forward']
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
#   %relu : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%primals_1,), kwargs = {})
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


# kernel path: inductor_cache/ew/cewedrdqkyiqzffhm4zixpngzun34dlvbk5y37v5qpgfooxwc5bm.py
# Topologically Sorted Source Nodes: [input_3, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => add_1, mul_1, mul_2, sub
#   x => relu_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 42)
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxvbz2nxg6h3hrw4x73nw4jtmokhpzropy3in3j4z7k7d7ruglyy.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_3 => add_3, mul_4, mul_5, sub_1
#   x_4 => relu_2
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 42)
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


# kernel path: inductor_cache/ma/cma733sxmqrl2jlrugrlozmzo5ygijn6idi6zyv44dy6kewb7xdm.py
# Topologically Sorted Source Nodes: [x_7, x_15, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_15 => add_9, mul_13, mul_14, sub_4
#   x_7 => add_5, mul_7, mul_8, sub_2
#   x_comb_iter_0 => add_10
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %add_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 42)
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


# kernel path: inductor_cache/d5/cd54ar4nczd2wed3imyrlfx4jpq5lntterbbjuqtzuhemlm2j27k.py
# Topologically Sorted Source Nodes: [x_comb_iter_1_left, x_23, x_comb_iter_1, x_comb_iter_2_left, x_31, x_comb_iter_2, x_comb_iter_3_right, x_comb_iter_3, x_32], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.avg_pool2d, aten.relu]
# Source node to ATen node mapping:
#   x_23 => add_14, mul_19, mul_20, sub_6
#   x_31 => add_19, mul_25, mul_26, sub_8
#   x_32 => relu_9
#   x_comb_iter_1 => add_15
#   x_comb_iter_1_left => _low_memory_max_pool2d_with_offsets, getitem_1
#   x_comb_iter_2 => add_20
#   x_comb_iter_2_left => avg_pool2d
#   x_comb_iter_3 => add_21
#   x_comb_iter_3_right => avg_pool2d_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_1, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %add_14), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_1, [3, 3], [2, 2], [1, 1], False, False), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool2d, %add_19), kwargs = {})
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_10, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool2d_1, %add_15), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 38, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x8 = xindex // 2
    x6 = xindex
    x4 = ((xindex // 4) % 42)
    x5 = xindex // 168
    x7 = (xindex % 168)
    tmp96 = tl.load(in_ptr1 + (x6), xmask)
    tmp97 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr6 + (x6), xmask)
    tmp114 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr8 + (x4), xmask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr9 + (x4), xmask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr11 + (x6), xmask)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x8), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x8), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x8), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x8), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 8*x8), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x8), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x8), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x8), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x8), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
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
    tmp77 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x8), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x8), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp77
    tmp80 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x8), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp79
    tmp82 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x8), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp82 + tmp81
    tmp84 = tl.load(in_ptr0 + (2*x0 + 8*x8), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 + tmp83
    tmp86 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x8), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 + tmp85
    tmp88 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x8), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp88 + tmp87
    tmp90 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x8), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp89
    tmp92 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x8), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp92 + tmp91
    tmp94 = ((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))) + ((4) * ((4) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (4)))*((4) * ((4) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (4))) + ((-1)*((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((4) * ((4) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (4)))) + ((-1)*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))*((4) * ((4) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (4))))
    tmp95 = tmp93 / tmp94
    tmp98 = tmp96 - tmp97
    tmp100 = 0.001
    tmp101 = tmp99 + tmp100
    tmp102 = libdevice.sqrt(tmp101)
    tmp103 = tl.full([1], 1, tl.int32)
    tmp104 = tmp103 / tmp102
    tmp105 = 1.0
    tmp106 = tmp104 * tmp105
    tmp107 = tmp98 * tmp106
    tmp109 = tmp107 * tmp108
    tmp111 = tmp109 + tmp110
    tmp112 = tmp51 + tmp111
    tmp115 = tmp113 - tmp114
    tmp117 = tmp116 + tmp100
    tmp118 = libdevice.sqrt(tmp117)
    tmp119 = tmp103 / tmp118
    tmp120 = tmp119 * tmp105
    tmp121 = tmp115 * tmp120
    tmp123 = tmp121 * tmp122
    tmp125 = tmp123 + tmp124
    tmp126 = tmp95 + tmp125
    tmp127 = (-1) + x1
    tmp128 = tmp127 >= tmp1
    tmp129 = tl.full([1], 2, tl.int64)
    tmp130 = tmp127 < tmp129
    tmp131 = tmp128 & tmp130
    tmp132 = (-1) + x0
    tmp133 = tmp132 >= tmp1
    tmp134 = tmp132 < tmp129
    tmp135 = tmp133 & tmp134
    tmp136 = tmp131 & tmp135
    tmp137 = tl.load(in_ptr11 + ((-3) + x6), tmp136 & xmask, other=0.0)
    tmp138 = x0
    tmp139 = tmp138 >= tmp1
    tmp140 = tmp138 < tmp129
    tmp141 = tmp139 & tmp140
    tmp142 = tmp131 & tmp141
    tmp143 = tl.load(in_ptr11 + ((-2) + x6), tmp142 & xmask, other=0.0)
    tmp144 = tmp143 + tmp137
    tmp145 = 1 + x0
    tmp146 = tmp145 >= tmp1
    tmp147 = tmp145 < tmp129
    tmp148 = tmp146 & tmp147
    tmp149 = tmp131 & tmp148
    tmp150 = tl.load(in_ptr11 + ((-1) + x6), tmp149 & xmask, other=0.0)
    tmp151 = tmp150 + tmp144
    tmp152 = x1
    tmp153 = tmp152 >= tmp1
    tmp154 = tmp152 < tmp129
    tmp155 = tmp153 & tmp154
    tmp156 = tmp155 & tmp135
    tmp157 = tl.load(in_ptr11 + ((-1) + x6), tmp156 & xmask, other=0.0)
    tmp158 = tmp157 + tmp151
    tmp159 = tmp155 & tmp141
    tmp160 = tl.load(in_ptr11 + (x6), tmp159 & xmask, other=0.0)
    tmp161 = tmp160 + tmp158
    tmp162 = tmp155 & tmp148
    tmp163 = tl.load(in_ptr11 + (1 + x6), tmp162 & xmask, other=0.0)
    tmp164 = tmp163 + tmp161
    tmp165 = 1 + x1
    tmp166 = tmp165 >= tmp1
    tmp167 = tmp165 < tmp129
    tmp168 = tmp166 & tmp167
    tmp169 = tmp168 & tmp135
    tmp170 = tl.load(in_ptr11 + (1 + x6), tmp169 & xmask, other=0.0)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp168 & tmp141
    tmp173 = tl.load(in_ptr11 + (2 + x6), tmp172 & xmask, other=0.0)
    tmp174 = tmp173 + tmp171
    tmp175 = tmp168 & tmp148
    tmp176 = tl.load(in_ptr11 + (3 + x6), tmp175 & xmask, other=0.0)
    tmp177 = tmp176 + tmp174
    tmp178 = 4 + ((-2)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) + ((-2)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp179 = tmp177 / tmp178
    tmp181 = tl.full([1], 0, tl.int32)
    tmp182 = triton_helpers.maximum(tmp181, tmp180)
    tmp183 = tmp179 + tmp112
    tl.store(out_ptr0 + (x6), tmp51, xmask)
    tl.store(out_ptr1 + (x6), tmp76, xmask)
    tl.store(out_ptr3 + (x7 + 672*x5), tmp112, xmask)
    tl.store(out_ptr4 + (x7 + 672*x5), tmp126, xmask)
    tl.store(out_ptr6 + (x6), tmp182, xmask)
    tl.store(out_ptr7 + (x7 + 672*x5), tmp183, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfp6yjw5p4mppyfhq7s6arfvrujfjgaeg6e3olvbs6m4f3er64f.py
# Topologically Sorted Source Nodes: [x_39, x_comb_iter_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_39 => add_25, mul_31, mul_32, sub_10
#   x_comb_iter_4 => add_26
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %getitem), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 42)
    x2 = xindex // 168
    x4 = (xindex % 168)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + 672*x2), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (42, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (42, ), (1, ))
    assert_size_stride(primals_4, (42, ), (1, ))
    assert_size_stride(primals_5, (42, ), (1, ))
    assert_size_stride(primals_6, (42, ), (1, ))
    assert_size_stride(primals_7, (42, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_8, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_9, (42, ), (1, ))
    assert_size_stride(primals_10, (42, ), (1, ))
    assert_size_stride(primals_11, (42, ), (1, ))
    assert_size_stride(primals_12, (42, ), (1, ))
    assert_size_stride(primals_13, (42, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_14, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_15, (42, ), (1, ))
    assert_size_stride(primals_16, (42, ), (1, ))
    assert_size_stride(primals_17, (42, ), (1, ))
    assert_size_stride(primals_18, (42, ), (1, ))
    assert_size_stride(primals_19, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_20, (42, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (42, ), (1, ))
    assert_size_stride(primals_22, (42, ), (1, ))
    assert_size_stride(primals_23, (42, ), (1, ))
    assert_size_stride(primals_24, (42, ), (1, ))
    assert_size_stride(primals_25, (42, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_26, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_27, (42, ), (1, ))
    assert_size_stride(primals_28, (42, ), (1, ))
    assert_size_stride(primals_29, (42, ), (1, ))
    assert_size_stride(primals_30, (42, ), (1, ))
    assert_size_stride(primals_31, (4, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_32, (42, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (42, ), (1, ))
    assert_size_stride(primals_34, (42, ), (1, ))
    assert_size_stride(primals_35, (42, ), (1, ))
    assert_size_stride(primals_36, (42, ), (1, ))
    assert_size_stride(primals_37, (42, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_38, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_39, (42, ), (1, ))
    assert_size_stride(primals_40, (42, ), (1, ))
    assert_size_stride(primals_41, (42, ), (1, ))
    assert_size_stride(primals_42, (42, ), (1, ))
    assert_size_stride(primals_43, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_44, (42, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_45, (42, ), (1, ))
    assert_size_stride(primals_46, (42, ), (1, ))
    assert_size_stride(primals_47, (42, ), (1, ))
    assert_size_stride(primals_48, (42, ), (1, ))
    assert_size_stride(primals_49, (42, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_51, (42, ), (1, ))
    assert_size_stride(primals_52, (42, ), (1, ))
    assert_size_stride(primals_53, (42, ), (1, ))
    assert_size_stride(primals_54, (42, ), (1, ))
    assert_size_stride(primals_55, (42, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_57, (42, ), (1, ))
    assert_size_stride(primals_58, (42, ), (1, ))
    assert_size_stride(primals_59, (42, ), (1, ))
    assert_size_stride(primals_60, (42, ), (1, ))
    assert_size_stride(primals_61, (42, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_62, (42, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_63, (42, ), (1, ))
    assert_size_stride(primals_64, (42, ), (1, ))
    assert_size_stride(primals_65, (42, ), (1, ))
    assert_size_stride(primals_66, (42, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(primals_1, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 42, 4, 4), (672, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 42, 4, 4), (672, 16, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 42, 4, 4), (672, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf1, primals_3, primals_4, primals_5, primals_6, buf2, buf3, 2688, grid=grid(2688), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_7, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf4, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 42, 2, 2), (168, 4, 2, 1))
        buf6 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, primals_9, primals_10, primals_11, primals_12, buf6, 672, grid=grid(672), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_13, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf7, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf0, primals_19, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf9, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 42, 2, 2), (168, 4, 2, 1))
        buf11 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf10, primals_21, primals_22, primals_23, primals_24, buf11, 672, grid=grid(672), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_25, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf12, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 42, 2, 2), (168, 4, 2, 1))
        buf14 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_15, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf8, primals_15, primals_16, primals_17, primals_18, buf13, primals_27, primals_28, primals_29, primals_30, buf14, 672, grid=grid(672), stream=stream0)
        del primals_18
        del primals_30
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf0, primals_31, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf17, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 42, 2, 2), (168, 4, 2, 1))
        buf19 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf18, primals_33, primals_34, primals_35, primals_36, buf19, 672, grid=grid(672), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_37, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf20, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf0, primals_43, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf24, (4, 4, 2, 2), (16, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 42, 2, 2), (168, 4, 2, 1))
        buf26 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf25, primals_45, primals_46, primals_47, primals_48, buf26, 672, grid=grid(672), stream=stream0)
        del primals_48
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_49, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf27, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 42, 2, 2), (168, 4, 2, 1))
        buf15 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.int8)
        buf39 = empty_strided_cuda((4, 168, 2, 2), (672, 4, 2, 1), torch.float32)
        buf22 = reinterpret_tensor(buf39, (4, 42, 2, 2), (672, 4, 2, 1), 0)  # alias
        buf36 = reinterpret_tensor(buf39, (4, 42, 2, 2), (672, 4, 2, 1), 168)  # alias
        buf30 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        buf37 = reinterpret_tensor(buf39, (4, 42, 2, 2), (672, 4, 2, 1), 336)  # alias
        # Topologically Sorted Source Nodes: [x_comb_iter_1_left, x_23, x_comb_iter_1, x_comb_iter_2_left, x_31, x_comb_iter_2, x_comb_iter_3_right, x_comb_iter_3, x_32], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.avg_pool2d, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4.run(buf2, buf21, primals_39, primals_40, primals_41, primals_42, buf28, primals_51, primals_52, primals_53, primals_54, buf14, buf15, buf16, buf22, buf36, buf30, buf37, 672, grid=grid(672), stream=stream0)
        del primals_42
        del primals_54
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf31, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 42, 2, 2), (168, 4, 2, 1))
        buf33 = empty_strided_cuda((4, 42, 2, 2), (168, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf32, primals_57, primals_58, primals_59, primals_60, buf33, 672, grid=grid(672), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf34, (4, 42, 2, 2), (168, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 42, 2, 2), (168, 4, 2, 1))
        buf38 = reinterpret_tensor(buf39, (4, 42, 2, 2), (672, 4, 2, 1), 504)  # alias
        # Topologically Sorted Source Nodes: [x_39, x_comb_iter_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_5.run(buf35, primals_63, primals_64, primals_65, primals_66, buf15, buf38, 672, grid=grid(672), stream=stream0)
        del buf15
        del primals_66
    return (buf39, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_19, primals_20, primals_21, primals_22, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_65, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf16, buf17, buf18, buf19, buf20, buf21, buf24, buf25, buf26, buf27, buf28, buf30, buf31, buf32, buf33, buf34, buf35, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((42, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((42, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((42, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((42, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((42, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((42, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((42, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((42, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((42, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((42, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((42, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((42, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
