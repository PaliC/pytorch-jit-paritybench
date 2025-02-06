# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/ev/cev7lvqum6mz46z5stmwb6vljaasq67yuydpbmr6o56jezfaof53.py
# Topologically Sorted Source Nodes: [conv2d, batch_norm, x_rgb_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   conv2d => convolution
#   x_rgb_1 => gt, mul_3, where
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%slice_2, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5a7xdzuu47p6z53sporaqnlg2azil7gmps6bxvr6y2grpgv6t5l.py
# Topologically Sorted Source Nodes: [x_rgb_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_rgb_2 => getitem, getitem_1
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ls/clsd6obiswd3rjurzkabwvzw3nkta262ncfk32zyrquh24t4jqzm.py
# Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, x_rgb_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   conv2d_1 => convolution_1
#   x_rgb_3 => gt_1, mul_7, where_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 48)
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/i2/ci2isylvvv6cysrcj5urkuemscrkcs3adw5nrus7wcwnellpqe3r.py
# Topologically Sorted Source Nodes: [x_rgb_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_rgb_4 => getitem_2, getitem_3
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: inductor_cache/ve/cveigndwbnjtnxhwhuoguodorfoosbfb67cwkdmz2o2n46j5yj7s.py
# Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, x_rgb_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_13, mul_14, sub_3
#   conv2d_3 => convolution_3
#   x_rgb_5 => gt_3, mul_15, where_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.2), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_7, %mul_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 48)
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wzp5ui66vgs4lj4lbymsfozy3oge55dujoj4lpikkoi6cyxo4h.py
# Topologically Sorted Source Nodes: [x_rgb_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_rgb_6 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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


# kernel path: inductor_cache/i2/ci2ovnyvwyi27cjdwjr2omhpky42ri7gprxmsi2xby3xyclq5gkv.py
# Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_5 => add_11, mul_21, mul_22, sub_5
#   conv2d_5 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 48)
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xp/cxp2r2hm3x4d67gzkzzmxqlplxvsvbxijbfmx4dr5gb2oa6s5fy7.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_5, %where_6], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 96)
    x0 = (xindex % 64)
    x2 = xindex // 6144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 3072*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 96, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x0 + 64*((-48) + x1) + 3072*x2), tmp13, other=0.0)
    tmp17 = 0.0
    tmp18 = tmp16 > tmp17
    tmp19 = 0.2
    tmp20 = tmp16 * tmp19
    tmp21 = tl.where(tmp18, tmp16, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tl.store(out_ptr0 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22j5ip2ibe3laur4wwnpdbhilkxxptrxkyqz42ormgpbenrc5hb.py
# Topologically Sorted Source Nodes: [x_rgb_7], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_rgb_7 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
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
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
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


# kernel path: inductor_cache/gp/cgp5zq35wie4gmuaeh4cnhprcilbqh24ggiynqjb2v3j5lkwloh2.py
# Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_11 => add_23, mul_45, mul_46, sub_11
#   conv2d_11 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_68, %primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_93), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %unsqueeze_95), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 48)
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4w/c4wr5xieyfil6w2ly2th5wysqjqo33bf5t66juq7zk4qtbuyxock.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_11, %where_12], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 96)
    x0 = (xindex % 16)
    x2 = xindex // 1536
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 768*x2), tmp4 & xmask, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 96, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x0 + 16*((-48) + x1) + 768*x2), tmp13 & xmask, other=0.0)
    tmp17 = 0.0
    tmp18 = tmp16 > tmp17
    tmp19 = 0.2
    tmp20 = tmp16 * tmp19
    tmp21 = tl.where(tmp18, tmp16, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chw7dhqjjpsykowoliea43zvbxszisgmhpbyqtmesej7r4zhnonu.py
# Topologically Sorted Source Nodes: [conv2d_18, batch_norm_18, x_inf_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm_18 => add_37, mul_73, mul_74, sub_18
#   conv2d_18 => convolution_18
#   x_inf_3 => gt_18, mul_75, where_18
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_110, %primals_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_149), kwargs = {})
#   %add_37 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_151), kwargs = {})
#   %gt_18 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_37, 0), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, 0.2), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %add_37, %mul_75), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjjimw3p5vyy6durwlfrokbstvhst7k7lqectx6g4s3wbe3z4iy.py
# Topologically Sorted Source Nodes: [x_inf_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_inf_4 => getitem_10, getitem_11
# Graph fragment:
#   %getitem_10 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_5, 0), kwargs = {})
#   %getitem_11 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_5, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_12 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_12(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/nz/cnzojzd4tadqnlqupu5hnejaqeb5zpimrxln57jpt65jjvdlnl7r.py
# Topologically Sorted Source Nodes: [conv2d_20, batch_norm_20, x_inf_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm_20 => add_41, mul_81, mul_82, sub_20
#   conv2d_20 => convolution_20
#   x_inf_5 => gt_20, mul_83, where_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_122, %primals_123, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %unsqueeze_165), kwargs = {})
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %unsqueeze_167), kwargs = {})
#   %gt_20 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_41, 0), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, 0.2), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %add_41, %mul_83), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/hz/chzu3ogiqaymk65pkoamu6duj67sjoboxbitgwmmberl435q44ov.py
# Topologically Sorted Source Nodes: [x_inf_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_inf_6 => getitem_12, getitem_13
# Graph fragment:
#   %getitem_12 : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_6, 0), kwargs = {})
#   %getitem_13 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_6, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_14 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_14(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pk/cpkgvjnvxxufquhdjya3cbs3wxkc3ljdqzuswx75pvh56fqhruoh.py
# Topologically Sorted Source Nodes: [conv2d_22, batch_norm_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_22 => add_45, mul_89, mul_90, sub_22
#   conv2d_22 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_134, %primals_135, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_181), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_183), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 18)
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7q/c7qszdkhr6kug3p742uqdk2wjerquij4mb7jnxeczjetam2imsas.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_6 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_22, %where_23], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 36)
    x0 = (xindex % 64)
    x2 = xindex // 2304
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 1152*x2), tmp4 & xmask, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 36, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x0 + 64*((-18) + x1) + 1152*x2), tmp13 & xmask, other=0.0)
    tmp17 = 0.0
    tmp18 = tmp16 > tmp17
    tmp19 = 0.2
    tmp20 = tmp16 * tmp19
    tmp21 = tl.where(tmp18, tmp16, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci36hbqjt7n6cdws2fbq46iqm6mwanmkwui6qwrklcniorkwzs3v.py
# Topologically Sorted Source Nodes: [x_inf_7], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_inf_7 => getitem_14, getitem_15
# Graph fragment:
#   %getitem_14 : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_7, 0), kwargs = {})
#   %getitem_15 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_7, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_17 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_17(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
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


# kernel path: inductor_cache/64/c64e3wyixuzlzjeng6uytxepya4pnwviinwo44yjp7h3b6j5l2ko.py
# Topologically Sorted Source Nodes: [conv2d_28, batch_norm_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_28 => add_57, mul_113, mul_114, sub_28
#   conv2d_28 => convolution_28
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %primals_170, %primals_171, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_229), kwargs = {})
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_231), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 18)
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
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ue/cuesxvui72dfimsxmckawkvcqpt7xmla7gjqkihmxl6o7r72xil4.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_9 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_28, %where_29], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 36)
    x0 = (xindex % 16)
    x2 = xindex // 576
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 288*x2), tmp4 & xmask, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 36, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x0 + 16*((-18) + x1) + 288*x2), tmp13 & xmask, other=0.0)
    tmp17 = 0.0
    tmp18 = tmp16 > tmp17
    tmp19 = 0.2
    tmp20 = tmp16 * tmp19
    tmp21 = tl.where(tmp18, tmp16, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6t/c6tmfrahqa5qgqdhs7qdtz6hqmanetjaq7bucx3clpoa7frj65bq.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_12 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_5, %cat_11], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 132)
    x0 = (xindex % 16)
    x2 = xindex // 2112
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 48, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 16*(x1) + 768*x2), tmp10 & xmask, other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp10, tmp16, tmp17)
    tmp19 = tmp5 >= tmp8
    tmp20 = tl.full([1], 96, tl.int64)
    tmp21 = tmp5 < tmp20
    tmp22 = tmp19 & tmp4
    tmp23 = tl.load(in_ptr1 + (x0 + 16*((-48) + (x1)) + 768*x2), tmp22 & xmask, other=0.0)
    tmp24 = 0.0
    tmp25 = tmp23 > tmp24
    tmp26 = 0.2
    tmp27 = tmp23 * tmp26
    tmp28 = tl.where(tmp25, tmp23, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp22, tmp28, tmp29)
    tmp31 = tl.where(tmp9, tmp18, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 132, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = (-96) + x1
    tmp38 = tl.full([1], 0, tl.int64)
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.full([1], 18, tl.int64)
    tmp41 = tmp37 < tmp40
    tmp42 = tmp41 & tmp34
    tmp43 = tl.load(in_ptr2 + (x0 + 16*((-96) + x1) + 288*x2), tmp42 & xmask, other=0.0)
    tmp44 = 0.0
    tmp45 = tmp43 > tmp44
    tmp46 = 0.2
    tmp47 = tmp43 * tmp46
    tmp48 = tl.where(tmp45, tmp43, tmp47)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp42, tmp48, tmp49)
    tmp51 = tmp37 >= tmp40
    tmp52 = tl.full([1], 36, tl.int64)
    tmp53 = tmp37 < tmp52
    tmp54 = tmp51 & tmp34
    tmp55 = tl.load(in_ptr3 + (x0 + 16*((-18) + ((-96) + x1)) + 288*x2), tmp54 & xmask, other=0.0)
    tmp56 = 0.0
    tmp57 = tmp55 > tmp56
    tmp58 = 0.2
    tmp59 = tmp55 * tmp58
    tmp60 = tl.where(tmp57, tmp55, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp54, tmp60, tmp61)
    tmp63 = tl.where(tmp41, tmp50, tmp62)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp34, tmp63, tmp64)
    tmp66 = tl.where(tmp4, tmp33, tmp65)
    tl.store(out_ptr0 + (x3), tmp66, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/au/caubqnfeh5gkbrojeyec6yfwdrzutwu2om6qewz66kk4kwfqni3y.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_13 => add_68, add_69, convert_element_type_68, convert_element_type_69, iota, mul_136, mul_137
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, 0), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_68, torch.float32), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_68, 0.0), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_69, 0.5), kwargs = {})
#   %convert_element_type_69 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_137, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_21 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/tp/ctpkr4aklheogd7tcdqxohd43p3ebbd7ctw6btiiqie3wyvhmn4p.py
# Topologically Sorted Source Nodes: [x_13, cat_13, add], Original ATen: [aten._unsafe_index, aten.cat, aten.add]
# Source node to ATen node mapping:
#   add => add_72
#   cat_13 => cat_13
#   x_13 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%cat_12, [None, None, %unsqueeze_272, %convert_element_type_69]), kwargs = {})
#   %cat_13 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %cat_8], 1), kwargs = {})
#   %add_72 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %cat_13), kwargs = {})
triton_poi_fused__unsafe_index_add_cat_22 = async_compile.triton('triton_poi_fused__unsafe_index_add_cat_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex // 64
    x2 = ((xindex // 64) % 132)
    x3 = xindex // 8448
    x4 = (xindex % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 4*tmp4 + 16*x5), xmask, eviction_policy='evict_last')
    tmp10 = x2
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 96, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 64*(x2) + 6144*x3), tmp14 & xmask, other=0.0)
    tmp16 = tmp10 >= tmp13
    tmp17 = tl.full([1], 132, tl.int64)
    tmp18 = tmp10 < tmp17
    tmp19 = tl.load(in_ptr3 + (x4 + 64*((-96) + x2) + 2304*x3), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tmp9 + tmp20
    tl.store(out_ptr0 + (x6), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b6/cb6fxfsh56e7x3glds4avoaiyxyl43gegncmsabhbci3vgkibvxb.py
# Topologically Sorted Source Nodes: [conv2d_34, batch_norm_34], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_34 => add_74, mul_141, mul_142, sub_34
#   conv2d_34 => convolution_34
# Graph fragment:
#   %convolution_34 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_72, %primals_206, %primals_207, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_274), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_276), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_141, %unsqueeze_278), kwargs = {})
#   %add_74 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_142, %unsqueeze_280), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/n3/cn3jj47pm3k4rnzbyyghooevanevwmtegb7yvdfx374pkp2hnr35.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_15 => add_75, add_76, convert_element_type_74, convert_element_type_75, iota_2, mul_144, mul_145
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, 0), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_75, torch.float32), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_74, 0.0), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, 0.5), kwargs = {})
#   %convert_element_type_75 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_145, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_24 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/dy/cdykkn5hopi42so23zatyrrdcqqncoyslqgz272msl6ha4uxawfe.py
# Topologically Sorted Source Nodes: [x_14, x_15, cat_14, add_1], Original ATen: [aten.leaky_relu, aten._unsafe_index, aten.cat, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_79
#   cat_14 => cat_14
#   x_14 => gt_34, mul_143, where_34
#   x_15 => _unsafe_index_1
# Graph fragment:
#   %gt_34 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_74, 0), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, 0.2), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %add_74, %mul_143), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_34, [None, None, %unsqueeze_281, %convert_element_type_75]), kwargs = {})
#   %cat_14 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%where_4, %where_21], 1), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_1, %cat_14), kwargs = {})
triton_poi_fused__unsafe_index_add_cat_leaky_relu_25 = async_compile.triton('triton_poi_fused__unsafe_index_add_cat_leaky_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_cat_leaky_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_cat_leaky_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex // 256
    x2 = ((xindex // 256) % 64)
    x3 = xindex // 16384
    x4 = (xindex % 256)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4 + 64*x5), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.2
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = x2
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 48, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tl.load(in_ptr2 + (x4 + 256*(x2) + 12288*x3), tmp19, other=0.0)
    tmp21 = tmp15 >= tmp18
    tmp22 = tl.full([1], 64, tl.int64)
    tmp23 = tmp15 < tmp22
    tmp24 = tl.load(in_ptr3 + (x4 + 256*((-48) + x2) + 4096*x3), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tmp14 + tmp25
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdoy3utbb5o2aldx4ttjznhbo65a4mfxcxb55dk7st6s2ufaj34d.py
# Topologically Sorted Source Nodes: [conv2d_35, batch_norm_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_35 => add_81, mul_149, mul_150, sub_35
#   conv2d_35 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_79, %primals_212, %primals_213, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_283), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_285), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %unsqueeze_287), kwargs = {})
#   %add_81 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_289), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/47/c47kyi4z3r3hcw5a3fptocsjolq4n4egx6fn3244czrdnyl5kdbd.py
# Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_17 => add_82, add_83, convert_element_type_80, convert_element_type_81, iota_4, mul_152, mul_153
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_152, 0), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_82, torch.float32), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_80, 0.0), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_83, 0.5), kwargs = {})
#   %convert_element_type_81 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_153, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_27 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_27(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gs/cgsk6swswzs3rixnaowgnd6c6xnvdkkugnznf63dqr2rklrjbup3.py
# Topologically Sorted Source Nodes: [x_16, x_17, cat_15, add_2], Original ATen: [aten.leaky_relu, aten._unsafe_index, aten.cat, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_86
#   cat_15 => cat_15
#   x_16 => gt_35, mul_151, where_35
#   x_17 => _unsafe_index_2
# Graph fragment:
#   %gt_35 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_81, 0), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, 0.2), kwargs = {})
#   %where_35 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %add_81, %mul_151), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_35, [None, None, %unsqueeze_290, %convert_element_type_81]), kwargs = {})
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%where_2, %where_19], 1), kwargs = {})
#   %add_86 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %cat_15), kwargs = {})
triton_poi_fused__unsafe_index_add_cat_leaky_relu_28 = async_compile.triton('triton_poi_fused__unsafe_index_add_cat_leaky_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_cat_leaky_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_cat_leaky_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex // 1024
    x2 = ((xindex // 1024) % 64)
    x3 = xindex // 65536
    x4 = (xindex % 1024)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*x5), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.2
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = x2
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 48, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tl.load(in_ptr2 + (x4 + 1024*(x2) + 49152*x3), tmp19, other=0.0)
    tmp21 = tmp15 >= tmp18
    tmp22 = tl.full([1], 64, tl.int64)
    tmp23 = tmp15 < tmp22
    tmp24 = tl.load(in_ptr3 + (x4 + 1024*((-48) + x2) + 16384*x3), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tmp14 + tmp25
    tl.store(out_ptr0 + (x6), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crx4rb2p6jupqqlvzyrjaokvw6umtokuezvz7yfnrzfvudf2isky.py
# Topologically Sorted Source Nodes: [conv2d_36, batch_norm_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_36 => add_88, mul_157, mul_158, sub_36
#   conv2d_36 => convolution_36
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_86, %primals_218, %primals_219, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_292), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_294), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_296), kwargs = {})
#   %add_88 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_298), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cdd7ft42xsa3424xjhzpwwesamqmongjv3mbtwedo65fbeiiav2l.py
# Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_19 => add_89, add_90, convert_element_type_86, convert_element_type_87, iota_6, mul_160, mul_161
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, 0), kwargs = {})
#   %convert_element_type_86 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_89, torch.float32), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_86, 0.0), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_90, 0.5), kwargs = {})
#   %convert_element_type_87 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_161, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_30 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_30(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/k7/ck7g2wbk3v3w444jxbvdi5hix52dhb2dasgxtwzlfb4ddg2lblfy.py
# Topologically Sorted Source Nodes: [x_18, x_19], Original ATen: [aten.leaky_relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   x_18 => gt_36, mul_159, where_36
#   x_19 => _unsafe_index_3
# Graph fragment:
#   %gt_36 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_88, 0), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_88, 0.2), kwargs = {})
#   %where_36 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_36, %add_88, %mul_159), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_36, [None, None, %unsqueeze_299, %convert_element_type_87]), kwargs = {})
triton_poi_fused__unsafe_index_leaky_relu_31 = async_compile.triton('triton_poi_fused__unsafe_index_leaky_relu_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_leaky_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_leaky_relu_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.2
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/6r/c6rfi7yp6agsyxccwaz4rjbr6ptg6p5rs75ovyrsrrfulrox64va.py
# Topologically Sorted Source Nodes: [conv2d_37, batch_norm_37, x_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   batch_norm_37 => add_94, mul_165, mul_166, sub_37
#   conv2d_37 => convolution_37
#   x_20 => gt_37, mul_167, where_37
# Graph fragment:
#   %convolution_37 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_224, %primals_225, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_301), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_303), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %unsqueeze_305), kwargs = {})
#   %add_94 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %unsqueeze_307), kwargs = {})
#   %gt_37 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_94, 0), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, 0.2), kwargs = {})
#   %where_37 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %add_94, %mul_167), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_32(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.2
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (48, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_9, (48, ), (1, ))
    assert_size_stride(primals_10, (48, ), (1, ))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_12, (48, ), (1, ))
    assert_size_stride(primals_13, (48, ), (1, ))
    assert_size_stride(primals_14, (48, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_15, (48, ), (1, ))
    assert_size_stride(primals_16, (48, ), (1, ))
    assert_size_stride(primals_17, (48, ), (1, ))
    assert_size_stride(primals_18, (48, ), (1, ))
    assert_size_stride(primals_19, (48, ), (1, ))
    assert_size_stride(primals_20, (48, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_21, (48, ), (1, ))
    assert_size_stride(primals_22, (48, ), (1, ))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_24, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_26, (48, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_30, (48, ), (1, ))
    assert_size_stride(primals_31, (48, ), (1, ))
    assert_size_stride(primals_32, (48, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_33, (48, ), (1, ))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (48, ), (1, ))
    assert_size_stride(primals_36, (48, ), (1, ))
    assert_size_stride(primals_37, (48, ), (1, ))
    assert_size_stride(primals_38, (48, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_39, (48, ), (1, ))
    assert_size_stride(primals_40, (48, ), (1, ))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_42, (48, ), (1, ))
    assert_size_stride(primals_43, (48, ), (1, ))
    assert_size_stride(primals_44, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_45, (48, ), (1, ))
    assert_size_stride(primals_46, (48, ), (1, ))
    assert_size_stride(primals_47, (48, ), (1, ))
    assert_size_stride(primals_48, (48, ), (1, ))
    assert_size_stride(primals_49, (48, ), (1, ))
    assert_size_stride(primals_50, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_51, (48, ), (1, ))
    assert_size_stride(primals_52, (48, ), (1, ))
    assert_size_stride(primals_53, (48, ), (1, ))
    assert_size_stride(primals_54, (48, ), (1, ))
    assert_size_stride(primals_55, (48, ), (1, ))
    assert_size_stride(primals_56, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_57, (48, ), (1, ))
    assert_size_stride(primals_58, (48, ), (1, ))
    assert_size_stride(primals_59, (48, ), (1, ))
    assert_size_stride(primals_60, (48, ), (1, ))
    assert_size_stride(primals_61, (48, ), (1, ))
    assert_size_stride(primals_62, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_63, (48, ), (1, ))
    assert_size_stride(primals_64, (48, ), (1, ))
    assert_size_stride(primals_65, (48, ), (1, ))
    assert_size_stride(primals_66, (48, ), (1, ))
    assert_size_stride(primals_67, (48, ), (1, ))
    assert_size_stride(primals_68, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_69, (48, ), (1, ))
    assert_size_stride(primals_70, (48, ), (1, ))
    assert_size_stride(primals_71, (48, ), (1, ))
    assert_size_stride(primals_72, (48, ), (1, ))
    assert_size_stride(primals_73, (48, ), (1, ))
    assert_size_stride(primals_74, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_75, (48, ), (1, ))
    assert_size_stride(primals_76, (48, ), (1, ))
    assert_size_stride(primals_77, (48, ), (1, ))
    assert_size_stride(primals_78, (48, ), (1, ))
    assert_size_stride(primals_79, (48, ), (1, ))
    assert_size_stride(primals_80, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_81, (48, ), (1, ))
    assert_size_stride(primals_82, (48, ), (1, ))
    assert_size_stride(primals_83, (48, ), (1, ))
    assert_size_stride(primals_84, (48, ), (1, ))
    assert_size_stride(primals_85, (48, ), (1, ))
    assert_size_stride(primals_86, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_87, (48, ), (1, ))
    assert_size_stride(primals_88, (48, ), (1, ))
    assert_size_stride(primals_89, (48, ), (1, ))
    assert_size_stride(primals_90, (48, ), (1, ))
    assert_size_stride(primals_91, (48, ), (1, ))
    assert_size_stride(primals_92, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_93, (48, ), (1, ))
    assert_size_stride(primals_94, (48, ), (1, ))
    assert_size_stride(primals_95, (48, ), (1, ))
    assert_size_stride(primals_96, (48, ), (1, ))
    assert_size_stride(primals_97, (48, ), (1, ))
    assert_size_stride(primals_98, (48, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_99, (48, ), (1, ))
    assert_size_stride(primals_100, (48, ), (1, ))
    assert_size_stride(primals_101, (48, ), (1, ))
    assert_size_stride(primals_102, (48, ), (1, ))
    assert_size_stride(primals_103, (48, ), (1, ))
    assert_size_stride(primals_104, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (16, ), (1, ))
    assert_size_stride(primals_106, (16, ), (1, ))
    assert_size_stride(primals_107, (16, ), (1, ))
    assert_size_stride(primals_108, (16, ), (1, ))
    assert_size_stride(primals_109, (16, ), (1, ))
    assert_size_stride(primals_110, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (16, ), (1, ))
    assert_size_stride(primals_113, (16, ), (1, ))
    assert_size_stride(primals_114, (16, ), (1, ))
    assert_size_stride(primals_115, (16, ), (1, ))
    assert_size_stride(primals_116, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_117, (16, ), (1, ))
    assert_size_stride(primals_118, (16, ), (1, ))
    assert_size_stride(primals_119, (16, ), (1, ))
    assert_size_stride(primals_120, (16, ), (1, ))
    assert_size_stride(primals_121, (16, ), (1, ))
    assert_size_stride(primals_122, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_123, (16, ), (1, ))
    assert_size_stride(primals_124, (16, ), (1, ))
    assert_size_stride(primals_125, (16, ), (1, ))
    assert_size_stride(primals_126, (16, ), (1, ))
    assert_size_stride(primals_127, (16, ), (1, ))
    assert_size_stride(primals_128, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_129, (16, ), (1, ))
    assert_size_stride(primals_130, (16, ), (1, ))
    assert_size_stride(primals_131, (16, ), (1, ))
    assert_size_stride(primals_132, (16, ), (1, ))
    assert_size_stride(primals_133, (16, ), (1, ))
    assert_size_stride(primals_134, (18, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_135, (18, ), (1, ))
    assert_size_stride(primals_136, (18, ), (1, ))
    assert_size_stride(primals_137, (18, ), (1, ))
    assert_size_stride(primals_138, (18, ), (1, ))
    assert_size_stride(primals_139, (18, ), (1, ))
    assert_size_stride(primals_140, (18, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_141, (18, ), (1, ))
    assert_size_stride(primals_142, (18, ), (1, ))
    assert_size_stride(primals_143, (18, ), (1, ))
    assert_size_stride(primals_144, (18, ), (1, ))
    assert_size_stride(primals_145, (18, ), (1, ))
    assert_size_stride(primals_146, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_147, (18, ), (1, ))
    assert_size_stride(primals_148, (18, ), (1, ))
    assert_size_stride(primals_149, (18, ), (1, ))
    assert_size_stride(primals_150, (18, ), (1, ))
    assert_size_stride(primals_151, (18, ), (1, ))
    assert_size_stride(primals_152, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_153, (18, ), (1, ))
    assert_size_stride(primals_154, (18, ), (1, ))
    assert_size_stride(primals_155, (18, ), (1, ))
    assert_size_stride(primals_156, (18, ), (1, ))
    assert_size_stride(primals_157, (18, ), (1, ))
    assert_size_stride(primals_158, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_159, (18, ), (1, ))
    assert_size_stride(primals_160, (18, ), (1, ))
    assert_size_stride(primals_161, (18, ), (1, ))
    assert_size_stride(primals_162, (18, ), (1, ))
    assert_size_stride(primals_163, (18, ), (1, ))
    assert_size_stride(primals_164, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_165, (18, ), (1, ))
    assert_size_stride(primals_166, (18, ), (1, ))
    assert_size_stride(primals_167, (18, ), (1, ))
    assert_size_stride(primals_168, (18, ), (1, ))
    assert_size_stride(primals_169, (18, ), (1, ))
    assert_size_stride(primals_170, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_171, (18, ), (1, ))
    assert_size_stride(primals_172, (18, ), (1, ))
    assert_size_stride(primals_173, (18, ), (1, ))
    assert_size_stride(primals_174, (18, ), (1, ))
    assert_size_stride(primals_175, (18, ), (1, ))
    assert_size_stride(primals_176, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_177, (18, ), (1, ))
    assert_size_stride(primals_178, (18, ), (1, ))
    assert_size_stride(primals_179, (18, ), (1, ))
    assert_size_stride(primals_180, (18, ), (1, ))
    assert_size_stride(primals_181, (18, ), (1, ))
    assert_size_stride(primals_182, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_183, (18, ), (1, ))
    assert_size_stride(primals_184, (18, ), (1, ))
    assert_size_stride(primals_185, (18, ), (1, ))
    assert_size_stride(primals_186, (18, ), (1, ))
    assert_size_stride(primals_187, (18, ), (1, ))
    assert_size_stride(primals_188, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_189, (18, ), (1, ))
    assert_size_stride(primals_190, (18, ), (1, ))
    assert_size_stride(primals_191, (18, ), (1, ))
    assert_size_stride(primals_192, (18, ), (1, ))
    assert_size_stride(primals_193, (18, ), (1, ))
    assert_size_stride(primals_194, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_195, (18, ), (1, ))
    assert_size_stride(primals_196, (18, ), (1, ))
    assert_size_stride(primals_197, (18, ), (1, ))
    assert_size_stride(primals_198, (18, ), (1, ))
    assert_size_stride(primals_199, (18, ), (1, ))
    assert_size_stride(primals_200, (18, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(primals_201, (18, ), (1, ))
    assert_size_stride(primals_202, (18, ), (1, ))
    assert_size_stride(primals_203, (18, ), (1, ))
    assert_size_stride(primals_204, (18, ), (1, ))
    assert_size_stride(primals_205, (18, ), (1, ))
    assert_size_stride(primals_206, (64, 132, 3, 3), (1188, 9, 3, 1))
    assert_size_stride(primals_207, (64, ), (1, ))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_213, (64, ), (1, ))
    assert_size_stride(primals_214, (64, ), (1, ))
    assert_size_stride(primals_215, (64, ), (1, ))
    assert_size_stride(primals_216, (64, ), (1, ))
    assert_size_stride(primals_217, (64, ), (1, ))
    assert_size_stride(primals_218, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (32, ), (1, ))
    assert_size_stride(primals_223, (32, ), (1, ))
    assert_size_stride(primals_224, (4, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_225, (4, ), (1, ))
    assert_size_stride(primals_226, (4, ), (1, ))
    assert_size_stride(primals_227, (4, ), (1, ))
    assert_size_stride(primals_228, (4, ), (1, ))
    assert_size_stride(primals_229, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (4, 3, 64, 64), (16384, 4096, 64, 1), 0), primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [conv2d, batch_norm, x_rgb_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0.run(buf1, buf3, primals_3, primals_4, primals_5, primals_6, primals_7, 262144, grid=grid(262144), stream=stream0)
        del primals_3
        buf4 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_rgb_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf3, buf4, buf5, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf4, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 48, 32, 32), (49152, 1024, 32, 1), torch.float32)
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, x_rgb_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2.run(buf7, buf9, primals_9, primals_10, primals_11, primals_12, primals_13, 196608, grid=grid(196608), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 48, 32, 32), (49152, 1024, 32, 1), torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, x_rgb_p2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2.run(buf11, buf13, primals_15, primals_16, primals_17, primals_18, primals_19, 196608, grid=grid(196608), stream=stream0)
        del primals_15
        buf14 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_rgb_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf13, buf14, buf15, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf14, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, x_rgb_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4.run(buf17, buf19, primals_21, primals_22, primals_23, primals_24, primals_25, 49152, grid=grid(49152), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4, x_rgb_p3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_4.run(buf21, buf23, primals_27, primals_28, primals_29, primals_30, primals_31, 49152, grid=grid(49152), stream=stream0)
        del primals_27
        buf24 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        buf25 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_rgb_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_5.run(buf23, buf24, buf25, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf27, primals_33, primals_34, primals_35, primals_36, primals_37, buf28, 12288, grid=grid(12288), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf24, primals_38, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_6, batch_norm_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf30, primals_39, primals_40, primals_41, primals_42, primals_43, buf31, 12288, grid=grid(12288), stream=stream0)
        del primals_39
        buf32 = empty_strided_cuda((4, 96, 8, 8), (6144, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf28, buf31, buf32, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7, batch_norm_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf34, primals_45, primals_46, primals_47, primals_48, primals_49, buf35, 12288, grid=grid(12288), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf32, primals_50, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf37 = buf36; del buf36  # reuse
        buf38 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [conv2d_8, batch_norm_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf37, primals_51, primals_52, primals_53, primals_54, primals_55, buf38, 12288, grid=grid(12288), stream=stream0)
        del primals_51
        buf39 = empty_strided_cuda((4, 96, 8, 8), (6144, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf35, buf38, buf39, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf41 = buf40; del buf40  # reuse
        buf42 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9, batch_norm_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf41, primals_57, primals_58, primals_59, primals_60, primals_61, buf42, 12288, grid=grid(12288), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf39, primals_62, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf44 = buf43; del buf43  # reuse
        buf45 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [conv2d_10, batch_norm_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_6.run(buf44, primals_63, primals_64, primals_65, primals_66, primals_67, buf45, 12288, grid=grid(12288), stream=stream0)
        del primals_63
        buf46 = empty_strided_cuda((4, 96, 8, 8), (6144, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf42, buf45, buf46, 24576, grid=grid(24576), stream=stream0)
        del buf42
        del buf45
        buf47 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.float32)
        buf48 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_rgb_7], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf46, buf47, buf48, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf47, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 48, 4, 4), (768, 16, 4, 1))
        buf50 = buf49; del buf49  # reuse
        buf51 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf50, primals_69, primals_70, primals_71, primals_72, primals_73, buf51, 3072, grid=grid(3072), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf47, primals_74, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 48, 4, 4), (768, 16, 4, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 48, 4, 4), (768, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_12, batch_norm_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf53, primals_75, primals_76, primals_77, primals_78, primals_79, buf54, 3072, grid=grid(3072), stream=stream0)
        del primals_75
        buf55 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf51, buf54, buf55, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 48, 4, 4), (768, 16, 4, 1))
        buf57 = buf56; del buf56  # reuse
        buf58 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [conv2d_13, batch_norm_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf57, primals_81, primals_82, primals_83, primals_84, primals_85, buf58, 3072, grid=grid(3072), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf55, primals_86, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 48, 4, 4), (768, 16, 4, 1))
        buf60 = buf59; del buf59  # reuse
        buf61 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [conv2d_14, batch_norm_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf60, primals_87, primals_88, primals_89, primals_90, primals_91, buf61, 3072, grid=grid(3072), stream=stream0)
        del primals_87
        buf62 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf58, buf61, buf62, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 48, 4, 4), (768, 16, 4, 1))
        buf64 = buf63; del buf63  # reuse
        buf65 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15, batch_norm_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf64, primals_93, primals_94, primals_95, primals_96, primals_97, buf65, 3072, grid=grid(3072), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf62, primals_98, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 48, 4, 4), (768, 16, 4, 1))
        buf67 = buf66; del buf66  # reuse
        buf68 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [conv2d_16, batch_norm_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf67, primals_99, primals_100, primals_101, primals_102, primals_103, buf68, 3072, grid=grid(3072), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(reinterpret_tensor(primals_1, (4, 1, 64, 64), (16384, 4096, 64, 1), 12288), primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf70 = buf69; del buf69  # reuse
        buf71 = empty_strided_cuda((4, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [conv2d_17, batch_norm_17, x_inf_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_0.run(buf70, buf72, primals_105, primals_106, primals_107, primals_108, primals_109, 262144, grid=grid(262144), stream=stream0)
        del primals_105
        buf73 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf74 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_inf_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf72, buf73, buf74, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf73, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf76 = buf75; del buf75  # reuse
        buf77 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [conv2d_18, batch_norm_18, x_inf_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11.run(buf76, buf78, primals_111, primals_112, primals_113, primals_114, primals_115, 65536, grid=grid(65536), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [conv2d_19, batch_norm_19, x_inf_p2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_11.run(buf80, buf82, primals_117, primals_118, primals_119, primals_120, primals_121, 65536, grid=grid(65536), stream=stream0)
        del primals_117
        buf83 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf84 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_inf_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_12.run(buf82, buf83, buf84, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf83, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf86 = buf85; del buf85  # reuse
        buf87 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [conv2d_20, batch_norm_20, x_inf_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13.run(buf86, buf88, primals_123, primals_124, primals_125, primals_126, primals_127, 16384, grid=grid(16384), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf90 = buf89; del buf89  # reuse
        buf91 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [conv2d_21, batch_norm_21, x_inf_p3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_13.run(buf90, buf92, primals_129, primals_130, primals_131, primals_132, primals_133, 16384, grid=grid(16384), stream=stream0)
        del primals_129
        buf93 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        buf94 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_inf_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_14.run(buf92, buf93, buf94, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf93, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 18, 8, 8), (1152, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_22, batch_norm_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf96, primals_135, primals_136, primals_137, primals_138, primals_139, buf97, 4608, grid=grid(4608), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf93, primals_140, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 18, 8, 8), (1152, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_23, batch_norm_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf99, primals_141, primals_142, primals_143, primals_144, primals_145, buf100, 4608, grid=grid(4608), stream=stream0)
        del primals_141
        buf101 = empty_strided_cuda((4, 36, 8, 8), (2304, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf97, buf100, buf101, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf103 = buf102; del buf102  # reuse
        buf104 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [conv2d_24, batch_norm_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf103, primals_147, primals_148, primals_149, primals_150, primals_151, buf104, 4608, grid=grid(4608), stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf101, primals_152, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf106 = buf105; del buf105  # reuse
        buf107 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [conv2d_25, batch_norm_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf106, primals_153, primals_154, primals_155, primals_156, primals_157, buf107, 4608, grid=grid(4608), stream=stream0)
        del primals_153
        buf108 = empty_strided_cuda((4, 36, 8, 8), (2304, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf104, buf107, buf108, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf110 = buf109; del buf109  # reuse
        buf111 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [conv2d_26, batch_norm_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf110, primals_159, primals_160, primals_161, primals_162, primals_163, buf111, 4608, grid=grid(4608), stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf108, primals_164, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 18, 8, 8), (1152, 64, 8, 1))
        buf113 = buf112; del buf112  # reuse
        buf114 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [conv2d_27, batch_norm_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_15.run(buf113, primals_165, primals_166, primals_167, primals_168, primals_169, buf114, 4608, grid=grid(4608), stream=stream0)
        del primals_165
        buf115 = empty_strided_cuda((4, 36, 8, 8), (2304, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf111, buf114, buf115, 9216, grid=grid(9216), stream=stream0)
        del buf111
        del buf114
        buf116 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        buf117 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_inf_7], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_17.run(buf115, buf116, buf117, 2304, grid=grid(2304), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf116, primals_170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 18, 4, 4), (288, 16, 4, 1))
        buf119 = buf118; del buf118  # reuse
        buf120 = empty_strided_cuda((4, 18, 4, 4), (288, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_28, batch_norm_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf119, primals_171, primals_172, primals_173, primals_174, primals_175, buf120, 1152, grid=grid(1152), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf116, primals_176, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 18, 4, 4), (288, 16, 4, 1))
        buf122 = buf121; del buf121  # reuse
        buf123 = empty_strided_cuda((4, 18, 4, 4), (288, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_29, batch_norm_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf122, primals_177, primals_178, primals_179, primals_180, primals_181, buf123, 1152, grid=grid(1152), stream=stream0)
        del primals_177
        buf124 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf120, buf123, buf124, 2304, grid=grid(2304), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 18, 4, 4), (288, 16, 4, 1))
        buf126 = buf125; del buf125  # reuse
        buf127 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [conv2d_30, batch_norm_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf126, primals_183, primals_184, primals_185, primals_186, primals_187, buf127, 1152, grid=grid(1152), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf124, primals_188, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 18, 4, 4), (288, 16, 4, 1))
        buf129 = buf128; del buf128  # reuse
        buf130 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [conv2d_31, batch_norm_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf129, primals_189, primals_190, primals_191, primals_192, primals_193, buf130, 1152, grid=grid(1152), stream=stream0)
        del primals_189
        buf131 = empty_strided_cuda((4, 36, 4, 4), (576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf127, buf130, buf131, 2304, grid=grid(2304), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 18, 4, 4), (288, 16, 4, 1))
        buf133 = buf132; del buf132  # reuse
        buf134 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [conv2d_32, batch_norm_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf133, primals_195, primals_196, primals_197, primals_198, primals_199, buf134, 1152, grid=grid(1152), stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf131, primals_200, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 18, 4, 4), (288, 16, 4, 1))
        buf136 = buf135; del buf135  # reuse
        buf137 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [conv2d_33, batch_norm_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_18.run(buf136, primals_201, primals_202, primals_203, primals_204, primals_205, buf137, 1152, grid=grid(1152), stream=stream0)
        del primals_201
        buf138 = empty_strided_cuda((4, 132, 4, 4), (2112, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf65, buf68, buf134, buf137, buf138, 8448, grid=grid(8448), stream=stream0)
        del buf134
        del buf137
        del buf65
        del buf68
        buf139 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_21.run(buf139, 8, grid=grid(8), stream=stream0)
        buf140 = empty_strided_cuda((4, 132, 8, 8), (8448, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, cat_13, add], Original ATen: [aten._unsafe_index, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_cat_22.run(buf139, buf138, buf46, buf115, buf140, 33792, grid=grid(33792), stream=stream0)
        del buf138
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf142 = buf141; del buf141  # reuse
        buf143 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_34, batch_norm_34], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_23.run(buf142, primals_207, primals_208, primals_209, primals_210, primals_211, buf143, 16384, grid=grid(16384), stream=stream0)
        del primals_207
        buf144 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_24.run(buf144, 16, grid=grid(16), stream=stream0)
        buf145 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15, cat_14, add_1], Original ATen: [aten.leaky_relu, aten._unsafe_index, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_cat_leaky_relu_25.run(buf144, buf143, buf23, buf92, buf145, 65536, grid=grid(65536), stream=stream0)
        del buf143
        # Topologically Sorted Source Nodes: [conv2d_35], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf147 = buf146; del buf146  # reuse
        buf148 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_35, batch_norm_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_26.run(buf147, primals_213, primals_214, primals_215, primals_216, primals_217, buf148, 65536, grid=grid(65536), stream=stream0)
        del primals_213
        buf149 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_27.run(buf149, 32, grid=grid(32), stream=stream0)
        buf150 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17, cat_15, add_2], Original ATen: [aten.leaky_relu, aten._unsafe_index, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_cat_leaky_relu_28.run(buf149, buf148, buf13, buf82, buf150, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf152 = buf151; del buf151  # reuse
        buf153 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_36, batch_norm_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_29.run(buf152, primals_219, primals_220, primals_221, primals_222, primals_223, buf153, 131072, grid=grid(131072), stream=stream0)
        del primals_219
        buf154 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_30.run(buf154, 64, grid=grid(64), stream=stream0)
        buf155 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18, x_19], Original ATen: [aten.leaky_relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_leaky_relu_31.run(buf154, buf153, buf155, 524288, grid=grid(524288), stream=stream0)
        del buf153
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf157 = buf156; del buf156  # reuse
        buf158 = reinterpret_tensor(buf148, (4, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf148  # reuse
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [conv2d_37, batch_norm_37, x_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_32.run(buf157, buf159, primals_225, primals_226, primals_227, primals_228, primals_229, 65536, grid=grid(65536), stream=stream0)
        del primals_225
    return (buf159, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_37, primals_38, primals_40, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_88, primals_89, primals_90, primals_91, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_121, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_157, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_176, primals_178, primals_179, primals_180, primals_181, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_190, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_206, primals_208, primals_209, primals_210, primals_211, primals_212, primals_214, primals_215, primals_216, primals_217, primals_218, primals_220, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, reinterpret_tensor(primals_1, (4, 3, 64, 64), (16384, 4096, 64, 1), 0), reinterpret_tensor(primals_1, (4, 1, 64, 64), (16384, 4096, 64, 1), 12288), buf1, buf3, buf4, buf5, buf7, buf9, buf11, buf13, buf14, buf15, buf17, buf19, buf21, buf23, buf24, buf25, buf27, buf30, buf32, buf34, buf37, buf39, buf41, buf44, buf46, buf47, buf48, buf50, buf53, buf55, buf57, buf60, buf62, buf64, buf67, buf70, buf72, buf73, buf74, buf76, buf78, buf80, buf82, buf83, buf84, buf86, buf88, buf90, buf92, buf93, buf94, buf96, buf99, buf101, buf103, buf106, buf108, buf110, buf113, buf115, buf116, buf117, buf119, buf122, buf124, buf126, buf129, buf131, buf133, buf136, buf139, buf140, buf142, buf144, buf145, buf147, buf149, buf150, buf152, buf154, buf155, buf157, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((48, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((48, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((48, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((48, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((18, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((18, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((18, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, 132, 3, 3), (1188, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((4, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
