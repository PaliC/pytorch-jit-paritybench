# AOT ID: ['30_forward']
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


# kernel path: inductor_cache/y3/cy3pkhwr7acowv2h7yizwlxz53ncfd6eav6c2xfp6kgsesjtl5ba.py
# Topologically Sorted Source Nodes: [out, out_1, x16, add, out16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.repeat, aten.add, aten.elu]
# Source node to ATen node mapping:
#   add => add_2
#   out => convolution
#   out16 => expm1, gt, mul_3, mul_5, where
#   out_1 => add_1, mul_1, mul_2, sub
#   x16 => repeat
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_8), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_11), kwargs = {})
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_3, [1, 16, 1, 1, 1]), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %repeat), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_2, 0), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_3, %mul_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_repeat_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_repeat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_repeat_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_repeat_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 16)
    x0 = (xindex % 262144)
    x2 = xindex // 4194304
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0 + 262144*x2), None, eviction_policy='evict_last')
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
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp22 = tmp19 * tmp11
    tmp23 = libdevice.expm1(tmp22)
    tmp24 = tmp23 * tmp11
    tmp25 = tl.where(tmp21, tmp22, tmp24)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6zbhxisc7zf3pmbrs6p2viekakxo6374vgo5g3tsb5gcvl6gdcl.py
# Topologically Sorted Source Nodes: [conv3d_1, batch_norm_1, down], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_4, mul_7, mul_8, sub_1
#   conv3d_1 => convolution_1
#   down => expm1_1, gt_1, mul_11, mul_9, where_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %primals_8, %primals_9, [2, 2, 2], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_14), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_17), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_20), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1.0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_9,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_1, 1.0), kwargs = {})
#   %where_1 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_9, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 32)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/bw/cbwhjdqqvgc52jwgryu62comdmqsy5n2dve4z23nkf2jgx3weokz.py
# Topologically Sorted Source Nodes: [conv3d_2, batch_norm_2, out_2, add_1, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_7
#   batch_norm_2 => add_6, mul_13, mul_14, sub_2
#   conv3d_2 => convolution_2
#   out_2 => expm1_2, gt_2, mul_15, mul_17, where_2
#   out_3 => expm1_3, gt_3, mul_18, mul_20, where_3
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %primals_14, %primals_15, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_26), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_29), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_32), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_35), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 0), kwargs = {})
#   %mul_15 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 1.0), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_15,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_2, 1.0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %mul_15, %mul_17), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_2, %where_1), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_18 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 1.0), kwargs = {})
#   %expm1_3 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_18,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_3, 1.0), kwargs = {})
#   %where_3 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mul_18, %mul_20), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/4f/c4fobmv73gtixlpxbhlq3mrxv6uwxal37sxvb3nvtzqio3yjaoka.py
# Topologically Sorted Source Nodes: [conv3d_3, batch_norm_3, down_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_9, mul_22, mul_23, sub_3
#   conv3d_3 => convolution_3
#   down_1 => expm1_4, gt_4, mul_24, mul_26, where_4
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %primals_20, %primals_21, [2, 2, 2], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_38), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_41), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_44), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_47), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 1.0), kwargs = {})
#   %expm1_4 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_24,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_4, 1.0), kwargs = {})
#   %where_4 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %mul_24, %mul_26), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/qw/cqw6hn4fcjwnp3wie73bmggjvbop5m6dexkqs7l5caq66oik5qlo.py
# Topologically Sorted Source Nodes: [conv3d_5, batch_norm_5, out_5, add_2, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_14
#   batch_norm_5 => add_13, mul_34, mul_35, sub_5
#   conv3d_5 => convolution_5
#   out_5 => expm1_6, gt_6, mul_36, mul_38, where_6
#   out_6 => expm1_7, gt_7, mul_39, mul_41, where_7
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_5, %primals_32, %primals_33, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_62), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_65), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_68), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_71), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %mul_36 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 1.0), kwargs = {})
#   %expm1_6 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_36,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_6, 1.0), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %mul_36, %mul_38), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %where_4), kwargs = {})
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_14, 0), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 1.0), kwargs = {})
#   %expm1_7 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_39,), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_7, 1.0), kwargs = {})
#   %where_7 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %mul_39, %mul_41), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/qd/cqdtrvq2wygvf3yts3d4kbuj36t37s2maq6se2rveryzxkdjyyoz.py
# Topologically Sorted Source Nodes: [conv3d_6, batch_norm_6, down_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_16, mul_43, mul_44, sub_6
#   conv3d_6 => convolution_6
#   down_2 => expm1_8, gt_8, mul_45, mul_47, where_8
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_7, %primals_38, %primals_39, [2, 2, 2], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_74), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_77), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_80), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_83), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_16, 0), kwargs = {})
#   %mul_45 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, 1.0), kwargs = {})
#   %expm1_8 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_45,), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_8, 1.0), kwargs = {})
#   %where_8 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %mul_45, %mul_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 128)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf37xkwhm6vr4onaqoahgj7v7payvugfgamd6gm6qxzzkhrbtljs.py
# Topologically Sorted Source Nodes: [conv3d_9, batch_norm_9, out_10, add_3, out_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_23
#   batch_norm_9 => add_22, mul_61, mul_62, sub_9
#   conv3d_9 => convolution_9
#   out_10 => expm1_11, gt_11, mul_63, mul_65, where_11
#   out_11 => expm1_12, gt_12, mul_66, mul_68, where_12
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_10, %primals_56, %primals_57, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_110), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_113), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_116), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_119), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_22, 0), kwargs = {})
#   %mul_63 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 1.0), kwargs = {})
#   %expm1_11 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_63,), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_11, 1.0), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %mul_63, %mul_65), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_11, %where_8), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_23, 0), kwargs = {})
#   %mul_66 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, 1.0), kwargs = {})
#   %expm1_12 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_66,), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_12, 1.0), kwargs = {})
#   %where_12 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %mul_66, %mul_68), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/hk/chke2hmx4u4yrrwvbjazhl3pzshxrjhwlf42bdfj42xo7lt5gwpe.py
# Topologically Sorted Source Nodes: [conv3d_10, batch_norm_10, down_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_10 => add_25, mul_70, mul_71, sub_10
#   conv3d_10 => convolution_10
#   down_3 => expm1_13, gt_13, mul_72, mul_74, where_13
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_12, %primals_62, %primals_63, [2, 2, 2], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_122), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_125), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_128), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_131), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_25, 0), kwargs = {})
#   %mul_72 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %expm1_13 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_72,), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_13, 1.0), kwargs = {})
#   %where_13 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %mul_72, %mul_74), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/cb/ccbsrxn4patbhow24zeirxhzbjfsws4nbn3vpwnnr6d7chbajmmr.py
# Topologically Sorted Source Nodes: [conv3d_12, batch_norm_12, out_14, add_4, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_30
#   batch_norm_12 => add_29, mul_82, mul_83, sub_12
#   conv3d_12 => convolution_12
#   out_14 => expm1_15, gt_15, mul_84, mul_86, where_15
#   out_15 => expm1_16, gt_16, mul_87, mul_89, where_16
# Graph fragment:
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_14, %primals_74, %primals_75, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_146), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_149), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_152), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_155), kwargs = {})
#   %gt_15 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_29, 0), kwargs = {})
#   %mul_84 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 1.0), kwargs = {})
#   %expm1_15 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_84,), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_15, 1.0), kwargs = {})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_15, %mul_84, %mul_86), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_15, %where_13), kwargs = {})
#   %gt_16 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_30, 0), kwargs = {})
#   %mul_87 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %expm1_16 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_87,), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_16, 1.0), kwargs = {})
#   %where_16 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_16, %mul_87, %mul_89), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/q5/cq54rzps7u374aebr2hxbrwdhfjaen67mqxa6prlq3b2rmjl5edh.py
# Topologically Sorted Source Nodes: [conv_transpose3d, batch_norm_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_13 => add_32, mul_91, mul_92, sub_13
#   conv_transpose3d => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_16, %primals_80, %primals_81, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_158), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_161), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_164), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_167), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 128)
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


# kernel path: inductor_cache/rf/crfyxch7eu6qitxps2t4eczf3uu35gyygc4yi6uajjhmg5xnpq4e.py
# Topologically Sorted Source Nodes: [xcat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xcat => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_17, %where_12], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 256)
    x0 = (xindex % 512)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 512*(x1) + 65536*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.expm1(tmp9)
    tmp11 = tmp10 * tmp8
    tmp12 = tl.where(tmp7, tmp9, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 256, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (x0 + 512*((-128) + x1) + 65536*x2), tmp15, other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceim2wnyvikn2zfkxflbv34bbs72smwlxyts65edojrmwgjmaiw4.py
# Topologically Sorted Source Nodes: [conv3d_13, batch_norm_14, out_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_14 => add_34, mul_97, mul_98, sub_14
#   conv3d_13 => convolution_14
#   out_18 => expm1_18, gt_18, mul_101, mul_99, where_18
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_86, %primals_87, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_170), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_173), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_176), kwargs = {})
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_179), kwargs = {})
#   %gt_18 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_34, 0), kwargs = {})
#   %mul_99 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 1.0), kwargs = {})
#   %expm1_18 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_99,), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_18, 1.0), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %mul_99, %mul_101), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 256)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck5ozwudr6saw7nw4hoky4jnmymbtoxow6tp3eo44wjknvsi4ufa.py
# Topologically Sorted Source Nodes: [conv3d_14, batch_norm_15, out_19, add_5, out_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_37
#   batch_norm_15 => add_36, mul_103, mul_104, sub_15
#   conv3d_14 => convolution_15
#   out_19 => expm1_19, gt_19, mul_105, mul_107, where_19
#   out_20 => expm1_20, gt_20, mul_108, mul_110, where_20
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %primals_92, %primals_93, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_182), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_185), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_188), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_191), kwargs = {})
#   %gt_19 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_36, 0), kwargs = {})
#   %mul_105 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, 1.0), kwargs = {})
#   %expm1_19 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_105,), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_19, 1.0), kwargs = {})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %mul_105, %mul_107), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_19, %cat), kwargs = {})
#   %gt_20 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_37, 0), kwargs = {})
#   %mul_108 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, 1.0), kwargs = {})
#   %expm1_20 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_108,), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_20, 1.0), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %mul_108, %mul_110), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjor3n4sghgpzzdrxnmgdshbow5n6higcukkwsnn5aid6tx6hxsq.py
# Topologically Sorted Source Nodes: [conv_transpose3d_1, batch_norm_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_16 => add_39, mul_112, mul_113, sub_16
#   conv_transpose3d_1 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_20, %primals_98, %primals_99, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_194), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_197), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_200), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_203), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/2d/c2dzuwmraihkolnsb4nxd4va2waddtrihugzgnh7xtzduymiv6ub.py
# Topologically Sorted Source Nodes: [xcat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xcat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_21, %where_7], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.expm1(tmp9)
    tmp11 = tmp10 * tmp8
    tmp12 = tl.where(tmp7, tmp9, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 128, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 262144*x2), tmp15, other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/zo/czotiicf7ir5w7ba75t5ms6kewjb2atltsdpqjtv62gr46jovsgl.py
# Topologically Sorted Source Nodes: [conv3d_15, batch_norm_17, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_17 => add_41, mul_118, mul_119, sub_17
#   conv3d_15 => convolution_17
#   out_23 => expm1_22, gt_22, mul_120, mul_122, where_22
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_104, %primals_105, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_206), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_209), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_212), kwargs = {})
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_215), kwargs = {})
#   %gt_22 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_41, 0), kwargs = {})
#   %mul_120 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, 1.0), kwargs = {})
#   %expm1_22 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_120,), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_22, 1.0), kwargs = {})
#   %where_22 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %mul_120, %mul_122), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 128)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/53/c53sesxgbwpnxoyd64yjsubvfy6jtc5pwjrmvvwdns3lbjhi7wge.py
# Topologically Sorted Source Nodes: [conv3d_16, batch_norm_18, out_24, add_6, out_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_44
#   batch_norm_18 => add_43, mul_124, mul_125, sub_18
#   conv3d_16 => convolution_18
#   out_24 => expm1_23, gt_23, mul_126, mul_128, where_23
#   out_25 => expm1_24, gt_24, mul_129, mul_131, where_24
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_22, %primals_110, %primals_111, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_218), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_221), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_224), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_227), kwargs = {})
#   %gt_23 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_43, 0), kwargs = {})
#   %mul_126 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, 1.0), kwargs = {})
#   %expm1_23 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_126,), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_23, 1.0), kwargs = {})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %mul_126, %mul_128), kwargs = {})
#   %add_44 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_23, %cat_1), kwargs = {})
#   %gt_24 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_44, 0), kwargs = {})
#   %mul_129 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_44, 1.0), kwargs = {})
#   %expm1_24 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_129,), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_24, 1.0), kwargs = {})
#   %where_24 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %mul_129, %mul_131), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/43/c43ovq357txfrkayg34vsiggfbar5na2hhbcvthir7nisbmkbk5n.py
# Topologically Sorted Source Nodes: [conv_transpose3d_2, batch_norm_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_19 => add_46, mul_133, mul_134, sub_19
#   conv_transpose3d_2 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_24, %primals_116, %primals_117, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_230), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_233), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_236), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 32)
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


# kernel path: inductor_cache/xd/cxdl2dlcuwulq6ciscp5qzcrcqxyap2ggqysxdwibapw47355jn7.py
# Topologically Sorted Source Nodes: [xcat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xcat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_25, %where_3], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32768) % 64)
    x0 = (xindex % 32768)
    x2 = xindex // 2097152
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 32768*(x1) + 1048576*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.expm1(tmp9)
    tmp11 = tmp10 * tmp8
    tmp12 = tl.where(tmp7, tmp9, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (x0 + 32768*((-32) + x1) + 1048576*x2), tmp15, other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ej/cejedg4j6oaz4qn5elhuiaojwdnjrkugkzrth2mmfktkucotyprv.py
# Topologically Sorted Source Nodes: [conv3d_17, batch_norm_20, out_27, add_7, out_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_7 => add_49
#   batch_norm_20 => add_48, mul_139, mul_140, sub_20
#   conv3d_17 => convolution_20
#   out_27 => expm1_26, gt_26, mul_141, mul_143, where_26
#   out_28 => expm1_27, gt_27, mul_144, mul_146, where_27
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %primals_122, %primals_123, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_242), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_245), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_248), kwargs = {})
#   %add_48 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_251), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_48, 0), kwargs = {})
#   %mul_141 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, 1.0), kwargs = {})
#   %expm1_26 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_141,), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_26, 1.0), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %mul_141, %mul_143), kwargs = {})
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_26, %cat_2), kwargs = {})
#   %gt_27 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_49, 0), kwargs = {})
#   %mul_144 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %expm1_27 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_144,), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_27, 1.0), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %mul_144, %mul_146), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_19(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenpovy2fwpnpwtzuayqcnf344ici74wxlpzi6pcjafaftvdjfsv.py
# Topologically Sorted Source Nodes: [conv_transpose3d_3, batch_norm_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_21 => add_51, mul_148, mul_149, sub_21
#   conv_transpose3d_3 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_27, %primals_128, %primals_129, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_254), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_257), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_260), kwargs = {})
#   %add_51 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 16)
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


# kernel path: inductor_cache/bu/cbu7zpajt6fy54uzohzibvk6lgyrnn4calxeb77ulo3ijz3so7wo.py
# Topologically Sorted Source Nodes: [xcat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xcat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%where_28, %where], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 262144) % 32)
    x0 = (xindex % 262144)
    x2 = xindex // 8388608
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 262144*(x1) + 4194304*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.expm1(tmp9)
    tmp11 = tmp10 * tmp8
    tmp12 = tl.where(tmp7, tmp9, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 32, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (x0 + 262144*((-16) + x1) + 4194304*x2), tmp15, other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/7p/c7piokfbsm42ilv7aqdi3ctrma5vj53mw4ha4cmzucpgihekyskg.py
# Topologically Sorted Source Nodes: [conv3d_18, batch_norm_22, out_30, add_8, out_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
# Source node to ATen node mapping:
#   add_8 => add_54
#   batch_norm_22 => add_53, mul_154, mul_155, sub_22
#   conv3d_18 => convolution_22
#   out_30 => expm1_29, gt_29, mul_156, mul_158, where_29
#   out_31 => expm1_30, gt_30, mul_159, mul_161, where_30
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_134, %primals_135, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_266), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_269), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_272), kwargs = {})
#   %add_53 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_275), kwargs = {})
#   %gt_29 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_53, 0), kwargs = {})
#   %mul_156 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 1.0), kwargs = {})
#   %expm1_29 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_156,), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_29, 1.0), kwargs = {})
#   %where_29 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %mul_156, %mul_158), kwargs = {})
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_29, %cat_3), kwargs = {})
#   %gt_30 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_54, 0), kwargs = {})
#   %mul_159 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, 1.0), kwargs = {})
#   %expm1_30 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_159,), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_30, 1.0), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %mul_159, %mul_161), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_22(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x3), None)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 > tmp18
    tmp27 = tmp25 * tmp11
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp11
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg6aaashqj67sjoxvwzv24to4ab5kmxcbkgie2qxfre4vscevoob.py
# Topologically Sorted Source Nodes: [conv3d_19, batch_norm_23, out_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   batch_norm_23 => add_56, mul_163, mul_164, sub_23
#   conv3d_19 => convolution_23
#   out_32 => expm1_31, gt_31, mul_165, mul_167, where_31
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_30, %primals_140, %primals_141, [1, 1, 1], [2, 2, 2], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_278), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_281), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_284), kwargs = {})
#   %add_56 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_287), kwargs = {})
#   %gt_31 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_56, 0), kwargs = {})
#   %mul_165 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, 1.0), kwargs = {})
#   %expm1_31 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_165,), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_31, 1.0), kwargs = {})
#   %where_31 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_31, %mul_165, %mul_167), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 4)
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
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5ywtregxmnxldze4b53ehjrygi3m76mcsyid5zzjn7llf4juhg.py
# Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_33 => convolution_24
# Graph fragment:
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_31, %primals_146, %primals_147, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 5, 5, 5), (125, 125, 25, 5, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (4, 1, 64, 64, 64), (262144, 262144, 4096, 64, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (32, 16, 2, 2, 2), (128, 8, 4, 2, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 32, 5, 5, 5), (4000, 125, 25, 5, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (64, 32, 2, 2, 2), (256, 8, 4, 2, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, 64, 5, 5, 5), (8000, 125, 25, 5, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 5, 5, 5), (8000, 125, 25, 5, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (128, 64, 2, 2, 2), (512, 8, 4, 2, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, 128, 5, 5, 5), (16000, 125, 25, 5, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, 128, 5, 5, 5), (16000, 125, 25, 5, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, 128, 5, 5, 5), (16000, 125, 25, 5, 1))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (256, 128, 2, 2, 2), (1024, 8, 4, 2, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, 256, 5, 5, 5), (32000, 125, 25, 5, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, 256, 5, 5, 5), (32000, 125, 25, 5, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, 128, 2, 2, 2), (1024, 8, 4, 2, 1))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (256, 256, 5, 5, 5), (32000, 125, 25, 5, 1))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, 256, 5, 5, 5), (32000, 125, 25, 5, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_98, (256, 64, 2, 2, 2), (512, 8, 4, 2, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, ), (1, ))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (128, 128, 5, 5, 5), (16000, 125, 25, 5, 1))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, 128, 5, 5, 5), (16000, 125, 25, 5, 1))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, 32, 2, 2, 2), (256, 8, 4, 2, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (64, 64, 5, 5, 5), (8000, 125, 25, 5, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_125, (64, ), (1, ))
    assert_size_stride(primals_126, (64, ), (1, ))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, 16, 2, 2, 2), (128, 8, 4, 2, 1))
    assert_size_stride(primals_129, (16, ), (1, ))
    assert_size_stride(primals_130, (16, ), (1, ))
    assert_size_stride(primals_131, (16, ), (1, ))
    assert_size_stride(primals_132, (16, ), (1, ))
    assert_size_stride(primals_133, (16, ), (1, ))
    assert_size_stride(primals_134, (32, 32, 5, 5, 5), (4000, 125, 25, 5, 1))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (32, ), (1, ))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (4, 32, 5, 5, 5), (4000, 125, 25, 5, 1))
    assert_size_stride(primals_141, (4, ), (1, ))
    assert_size_stride(primals_142, (4, ), (1, ))
    assert_size_stride(primals_143, (4, ), (1, ))
    assert_size_stride(primals_144, (4, ), (1, ))
    assert_size_stride(primals_145, (4, ), (1, ))
    assert_size_stride(primals_146, (4, 4, 1, 1, 1), (4, 1, 1, 1, 1))
    assert_size_stride(primals_147, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 64, 64, 64), (4194304, 262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 16, 64, 64, 64), (4194304, 262144, 4096, 64, 1), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, x16, add, out16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.repeat, aten.add, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_repeat_0.run(buf1, buf3, primals_2, primals_4, primals_5, primals_6, primals_7, primals_3, 16777216, grid=grid(16777216), stream=stream0)
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [conv3d_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_8, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1))
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cuda((4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [conv3d_1, batch_norm_1, down], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1.run(buf5, buf7, primals_9, primals_10, primals_11, primals_12, primals_13, 4194304, grid=grid(4194304), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [conv3d_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_14, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [conv3d_2, batch_norm_2, out_2, add_1, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_2.run(buf9, buf11, primals_15, primals_16, primals_17, primals_18, primals_19, buf7, 4194304, grid=grid(4194304), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [conv3d_3], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_20, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [conv3d_3, batch_norm_3, down_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3.run(buf13, buf15, primals_21, primals_22, primals_23, primals_24, primals_25, 1048576, grid=grid(1048576), stream=stream0)
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [conv3d_4], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_26, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [conv3d_4, batch_norm_4, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_3.run(buf17, buf19, primals_27, primals_28, primals_29, primals_30, primals_31, 1048576, grid=grid(1048576), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [conv3d_5], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_32, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [conv3d_5, batch_norm_5, out_5, add_2, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_4.run(buf21, buf23, primals_33, primals_34, primals_35, primals_36, primals_37, buf15, 1048576, grid=grid(1048576), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv3d_6], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_38, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [conv3d_6, batch_norm_6, down_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5.run(buf25, buf27, primals_39, primals_40, primals_41, primals_42, primals_43, 262144, grid=grid(262144), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [conv3d_7], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_44, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [conv3d_7, batch_norm_7, out_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5.run(buf29, buf31, primals_45, primals_46, primals_47, primals_48, primals_49, 262144, grid=grid(262144), stream=stream0)
        del primals_45
        del primals_49
        # Topologically Sorted Source Nodes: [conv3d_8], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_50, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [conv3d_8, batch_norm_8, out_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_5.run(buf33, buf35, primals_51, primals_52, primals_53, primals_54, primals_55, 262144, grid=grid(262144), stream=stream0)
        del primals_51
        del primals_55
        # Topologically Sorted Source Nodes: [conv3d_9], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_56, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf37 = buf36; del buf36  # reuse
        buf38 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [conv3d_9, batch_norm_9, out_10, add_3, out_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_6.run(buf37, buf39, primals_57, primals_58, primals_59, primals_60, primals_61, buf27, 262144, grid=grid(262144), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [conv3d_10], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_62, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 256, 4, 4, 4), (16384, 64, 16, 4, 1))
        buf41 = buf40; del buf40  # reuse
        buf42 = empty_strided_cuda((4, 256, 4, 4, 4), (16384, 64, 16, 4, 1), torch.float32)
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [conv3d_10, batch_norm_10, down_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7.run(buf41, buf43, primals_63, primals_64, primals_65, primals_66, primals_67, 65536, grid=grid(65536), stream=stream0)
        del primals_63
        del primals_67
        # Topologically Sorted Source Nodes: [conv3d_11], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_68, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 256, 4, 4, 4), (16384, 64, 16, 4, 1))
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((4, 256, 4, 4, 4), (16384, 64, 16, 4, 1), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [conv3d_11, batch_norm_11, out_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_7.run(buf45, buf47, primals_69, primals_70, primals_71, primals_72, primals_73, 65536, grid=grid(65536), stream=stream0)
        del primals_69
        del primals_73
        # Topologically Sorted Source Nodes: [conv3d_12], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_74, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 256, 4, 4, 4), (16384, 64, 16, 4, 1))
        buf49 = buf48; del buf48  # reuse
        buf50 = empty_strided_cuda((4, 256, 4, 4, 4), (16384, 64, 16, 4, 1), torch.float32)
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [conv3d_12, batch_norm_12, out_14, add_4, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_8.run(buf49, buf51, primals_75, primals_76, primals_77, primals_78, primals_79, buf43, 65536, grid=grid(65536), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [conv_transpose3d], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_80, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_transpose3d, batch_norm_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_9.run(buf53, primals_81, primals_82, primals_83, primals_84, primals_85, buf54, 262144, grid=grid(262144), stream=stream0)
        del primals_81
        buf55 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xcat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf54, buf39, buf55, 524288, grid=grid(524288), stream=stream0)
        del buf54
        # Topologically Sorted Source Nodes: [conv3d_13], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_86, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf57 = buf56; del buf56  # reuse
        buf58 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [conv3d_13, batch_norm_14, out_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_11.run(buf57, buf59, primals_87, primals_88, primals_89, primals_90, primals_91, 524288, grid=grid(524288), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [conv3d_14], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_92, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf61 = buf60; del buf60  # reuse
        buf62 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [conv3d_14, batch_norm_15, out_19, add_5, out_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_12.run(buf61, buf63, primals_93, primals_94, primals_95, primals_96, primals_97, buf55, 524288, grid=grid(524288), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [conv_transpose3d_1], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_98, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_transpose3d_1, batch_norm_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_13.run(buf65, primals_99, primals_100, primals_101, primals_102, primals_103, buf66, 1048576, grid=grid(1048576), stream=stream0)
        del primals_99
        buf67 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xcat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf66, buf23, buf67, 2097152, grid=grid(2097152), stream=stream0)
        del buf66
        # Topologically Sorted Source Nodes: [conv3d_15], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_104, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [conv3d_15, batch_norm_17, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_15.run(buf69, buf71, primals_105, primals_106, primals_107, primals_108, primals_109, 2097152, grid=grid(2097152), stream=stream0)
        del primals_105
        del primals_109
        # Topologically Sorted Source Nodes: [conv3d_16], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_110, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf73 = buf72; del buf72  # reuse
        buf74 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [conv3d_16, batch_norm_18, out_24, add_6, out_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_16.run(buf73, buf75, primals_111, primals_112, primals_113, primals_114, primals_115, buf67, 2097152, grid=grid(2097152), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [conv_transpose3d_2], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_116, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1))
        buf77 = buf76; del buf76  # reuse
        buf78 = empty_strided_cuda((4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_transpose3d_2, batch_norm_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_17.run(buf77, primals_117, primals_118, primals_119, primals_120, primals_121, buf78, 4194304, grid=grid(4194304), stream=stream0)
        del primals_117
        buf79 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xcat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf78, buf11, buf79, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [conv3d_17], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_122, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [conv3d_17, batch_norm_20, out_27, add_7, out_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_19.run(buf81, buf83, primals_123, primals_124, primals_125, primals_126, primals_127, buf79, 8388608, grid=grid(8388608), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [conv_transpose3d_3], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_128, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 16, 64, 64, 64), (4194304, 262144, 4096, 64, 1))
        buf85 = buf84; del buf84  # reuse
        buf86 = empty_strided_cuda((4, 16, 64, 64, 64), (4194304, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv_transpose3d_3, batch_norm_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_20.run(buf85, primals_129, primals_130, primals_131, primals_132, primals_133, buf86, 16777216, grid=grid(16777216), stream=stream0)
        del primals_129
        buf87 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xcat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf86, buf3, buf87, 33554432, grid=grid(33554432), stream=stream0)
        del buf86
        # Topologically Sorted Source Nodes: [conv3d_18], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_134, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [conv3d_18, batch_norm_22, out_30, add_8, out_31], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_elu_22.run(buf89, buf91, primals_135, primals_136, primals_137, primals_138, primals_139, buf87, 33554432, grid=grid(33554432), stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [conv3d_19], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_140, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 4, 64, 64, 64), (1048576, 262144, 4096, 64, 1))
        buf93 = buf92; del buf92  # reuse
        buf94 = reinterpret_tensor(buf78, (4, 4, 64, 64, 64), (1048576, 262144, 4096, 64, 1), 0); del buf78  # reuse
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [conv3d_19, batch_norm_23, out_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_23.run(buf93, buf95, primals_141, primals_142, primals_143, primals_144, primals_145, 4194304, grid=grid(4194304), stream=stream0)
        del primals_141
        del primals_145
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_146, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 4, 64, 64, 64), (1048576, 262144, 4096, 64, 1))
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf97, primals_147, 4194304, grid=grid(4194304), stream=stream0)
        del primals_147
    return (buf97, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_37, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_70, primals_71, primals_72, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_121, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_146, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf95, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 1, 5, 5, 5), (125, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 64, 64, 64), (262144, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 16, 2, 2, 2), (128, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 5, 5, 5), (4000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 32, 2, 2, 2), (256, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 5, 5, 5), (8000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 5, 5, 5), (8000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 64, 2, 2, 2), (512, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 128, 5, 5, 5), (16000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 128, 5, 5, 5), (16000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 5, 5, 5), (16000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 128, 2, 2, 2), (1024, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 256, 5, 5, 5), (32000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 256, 5, 5, 5), (32000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, 128, 2, 2, 2), (1024, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 256, 5, 5, 5), (32000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, 256, 5, 5, 5), (32000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, 64, 2, 2, 2), (512, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, 128, 5, 5, 5), (16000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, 128, 5, 5, 5), (16000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 32, 2, 2, 2), (256, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 64, 5, 5, 5), (8000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 16, 2, 2, 2), (128, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, 32, 5, 5, 5), (4000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((4, 32, 5, 5, 5), (4000, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((4, 4, 1, 1, 1), (4, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
