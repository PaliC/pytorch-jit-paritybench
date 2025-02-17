# AOT ID: ['23_forward']
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


# kernel path: inductor_cache/md/cmden36j7rnruht4do2cdt2eidbjl22wptacvgvd6jp3gcb6bcja.py
# Topologically Sorted Source Nodes: [batch_norm, output_1, batch_norm_1, output_2, output1_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   output1_add => add_4
#   output_1 => gt, mul_3, where
#   output_2 => gt_1, mul_7, where_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %add_1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %add_3), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %where), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 8)
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
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp28 > tmp29
    tmp32 = tmp31 * tmp28
    tmp33 = tl.where(tmp30, tmp28, tmp32)
    tmp34 = tmp15 > tmp29
    tmp36 = tmp35 * tmp15
    tmp37 = tl.where(tmp34, tmp15, tmp36)
    tmp38 = tmp33 + tmp37
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7ua7fikoalnwneh2hl37wy6mqoy6v26ibzuyg4d34r47o5eod6.py
# Topologically Sorted Source Nodes: [output_1, output_2, batch_norm_2, output_3, output1, batch_norm_3, output_5, long1, output1_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   batch_norm_2 => add_6, mul_10, mul_9, sub_2
#   batch_norm_3 => add_9, mul_13, mul_14, sub_3
#   long1 => add_10
#   output1 => add_7
#   output1_add_1 => add_11
#   output_1 => gt, mul_3, where
#   output_2 => gt_1, mul_7, where_1
#   output_3 => gt_2, mul_11, where_2
#   output_5 => gt_3, mul_15, where_3
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %add_1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %add_3), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_6), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_6, %mul_11), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_2, %where_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_9), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_9, %mul_15), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %where), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_10), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 8)
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
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x3), None)
    tmp46 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp34 > tmp29
    tmp37 = tmp36 * tmp34
    tmp38 = tl.where(tmp35, tmp34, tmp37)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp28 > tmp29
    tmp42 = tmp41 * tmp28
    tmp43 = tl.where(tmp40, tmp28, tmp42)
    tmp45 = tmp44 > tmp29
    tmp47 = tmp46 * tmp44
    tmp48 = tl.where(tmp45, tmp44, tmp47)
    tmp49 = tmp43 + tmp48
    tmp50 = tmp39 + tmp49
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciijpbzgqc65y6wcfew3jqqk4w3maccb4w23zdvfjwxb3efdxjnu.py
# Topologically Sorted Source Nodes: [output_2, output_3, output1, batch_norm_4, output_6, output1_1], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_4 => add_13, mul_17, mul_18, sub_4
#   output1 => add_7
#   output1_1 => add_14
#   output_2 => gt_1, mul_7, where_1
#   output_3 => gt_2, mul_11, where_2
#   output_6 => gt_4, mul_19, where_4
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %add_3), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_7), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_6), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_6, %mul_11), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_2, %where_1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %add_13), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_13, %mul_19), kwargs = {})
#   %add_14 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %add_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
    tmp28 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp26 > tmp16
    tmp29 = tmp28 * tmp26
    tmp30 = tl.where(tmp27, tmp26, tmp29)
    tmp31 = tmp25 + tmp30
    tmp32 = tmp20 + tmp31
    tl.store(in_out_ptr0 + (x3), tmp32, None)
''', device_str='cuda')


# kernel path: inductor_cache/k7/ck77liuz4s3gbesf2d67naqvzqumcjm7cu4kmwfael5n3qh32xqa.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_10, %add_14], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 16)
    x0 = (xindex % 1024)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 * tmp5
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.load(in_ptr2 + (x0 + 1024*(x1) + 8192*x2), tmp4, other=0.0)
    tmp12 = tmp11 > tmp6
    tmp13 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp11
    tmp15 = tl.where(tmp12, tmp11, tmp14)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 16, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + (x0 + 1024*((-8) + x1) + 8192*x2), tmp19, other=0.0)
    tmp23 = tl.where(tmp4, tmp18, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/kz/ckzulnir7hchuwnyjfmccg3xkamtj7ueumjzjxnp2f7s57nagum6.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_2 => add_17, mul_21, mul_22, sub_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 16)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/vb/cvbj2ufsxk7wc27gircfule4nhhe7z66xiqqe5rn7inysm5i4h3h.py
# Topologically Sorted Source Nodes: [output_1, output_5, long1, x1, add_6], Original ATen: [aten._prelu_kernel, aten.add, aten.clone]
# Source node to ATen node mapping:
#   add_6 => add_18
#   long1 => add_10
#   output_1 => gt, mul_3, where
#   output_5 => gt_3, mul_15, where_3
#   x1 => clone
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %add_1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_9), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_9, %mul_15), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %where), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone, %add_10), kwargs = {})
triton_poi_fused__prelu_kernel_add_clone_5 = async_compile.triton('triton_poi_fused__prelu_kernel_add_clone_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_clone_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_clone_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 8192
    x3 = (xindex % 8192)
    x4 = xindex
    x1 = ((xindex // 1024) % 8)
    tmp0 = tl.load(in_ptr0 + (x3 + 16384*x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tmp7 > tmp2
    tmp10 = tmp9 * tmp7
    tmp11 = tl.where(tmp8, tmp7, tmp10)
    tmp12 = tmp6 + tmp11
    tmp13 = tmp0 + tmp12
    tl.store(in_out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/5h/c5hsi4mnafuvee2erymuijkaspw3ttowifm6oawpmq2techjqdcq.py
# Topologically Sorted Source Nodes: [x2, add_7], Original ATen: [aten.clone, aten.add]
# Source node to ATen node mapping:
#   add_7 => add_21
#   x2 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_6,), kwargs = {memory_format: torch.contiguous_format})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_1, %add_14), kwargs = {})
triton_poi_fused_add_clone_6 = async_compile.triton('triton_poi_fused_add_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8192 + x0 + 16384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvafi2edisiw72repoo33bl2hkytp5kdzccmr5hyjjfeirflue7f.py
# Topologically Sorted Source Nodes: [p, d1_1], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_1 => add_22
#   p => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_10, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %avg_pool2d), kwargs = {})
triton_poi_fused_add_avg_pool2d_7 = async_compile.triton('triton_poi_fused_add_avg_pool2d_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x4 = xindex // 16
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33)))*((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33))) + ((-2)*x0*((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33)))) + ((-2)*x1*((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33)))) + 4*x0*x1 + ((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33))) + ((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o6/co6krfcvwji4sprki6d4a5rof4hf37uz5dqqzqd2tzqys7wsralc.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_22, %add_23, %add_24, %add_25], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 24)
    x0 = (xindex % 256)
    x2 = xindex // 6144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 1536*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 12, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 256*((-6) + x1) + 1536*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + 256*((-6) + x1) + 1536*x2), tmp9, other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp0 >= tmp7
    tmp16 = tl.full([1], 18, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + (x0 + 256*((-12) + x1) + 1536*x2), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr1 + (x0 + 256*((-12) + x1) + 1536*x2), tmp18, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr2 + (x0 + 256*((-12) + x1) + 1536*x2), tmp18, other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = tmp0 >= tmp16
    tmp27 = tl.full([1], 24, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 256*((-18) + x1) + 1536*x2), tmp26, other=0.0)
    tmp30 = tl.load(in_ptr1 + (x0 + 256*((-18) + x1) + 1536*x2), tmp26, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr2 + (x0 + 256*((-18) + x1) + 1536*x2), tmp26, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr3 + (x0 + 256*((-18) + x1) + 1536*x2), tmp26, other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp18, tmp25, tmp37)
    tmp39 = tl.where(tmp9, tmp14, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x3), tmp40, None)
''', device_str='cuda')


# kernel path: inductor_cache/xa/cxaeomutaktlasp3vppc6t3qv766uhj4zccelhjtqxa6lsxn43o4.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_26, %add_27, %add_28, %add_29], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 24)
    x0 = (xindex % 256)
    x2 = xindex // 6144
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 1536*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 1024*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 12, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (x0 + 256*((-6) + x1) + 1536*x2), tmp15, other=0.0)
    tmp17 = tl.load(in_ptr2 + (x0 + 256*((-6) + x1) + 1536*x2), tmp15, other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.load(in_ptr1 + (256 + x0 + 1024*x2), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tmp0 >= tmp13
    tmp26 = tl.full([1], 18, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 256*((-12) + x1) + 1536*x2), tmp28, other=0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + 256*((-12) + x1) + 1536*x2), tmp28, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr3 + (x0 + 256*((-12) + x1) + 1536*x2), tmp28, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr1 + (512 + x0 + 1024*x2), tmp28, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp28, tmp37, tmp38)
    tmp40 = tmp0 >= tmp26
    tmp41 = tl.full([1], 24, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tl.load(in_ptr0 + (x0 + 256*((-18) + x1) + 1536*x2), tmp40, other=0.0)
    tmp44 = tl.load(in_ptr2 + (x0 + 256*((-18) + x1) + 1536*x2), tmp40, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr3 + (x0 + 256*((-18) + x1) + 1536*x2), tmp40, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr4 + (x0 + 256*((-18) + x1) + 1536*x2), tmp40, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.load(in_ptr1 + (768 + x0 + 1024*x2), tmp40, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp49 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp40, tmp53, tmp54)
    tmp56 = tl.where(tmp28, tmp39, tmp55)
    tmp57 = tl.where(tmp15, tmp24, tmp56)
    tmp58 = tl.where(tmp4, tmp11, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/oa/coafhvhdksb7v33g67owa3utbtade5mho6sepmsp624s6gotcbvm.py
# Topologically Sorted Source Nodes: [batch_norm_6, output_8, batch_norm_7, output_11, output2_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_6 => add_20, mul_24, mul_25, sub_6
#   batch_norm_7 => add_31, mul_32, mul_33, sub_7
#   output2_add => add_32
#   output_11 => gt_6, mul_34, where_6
#   output_8 => gt_5, mul_26, where_5
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_49), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_53), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_55), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_20, %mul_26), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_61), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_63), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %unsqueeze_65), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %unsqueeze_67), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %add_31), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_31, %mul_34), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %where_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 24)
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
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp28 > tmp29
    tmp32 = tmp31 * tmp28
    tmp33 = tl.where(tmp30, tmp28, tmp32)
    tmp34 = tmp15 > tmp29
    tmp36 = tmp35 * tmp15
    tmp37 = tl.where(tmp34, tmp15, tmp36)
    tmp38 = tmp33 + tmp37
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrgbxrwzab3je4ebmyjpodhibphbgwedpk77budt3bgrhxcvc7p.py
# Topologically Sorted Source Nodes: [p_1, d1_4], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_4 => add_33
#   p_1 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_17, [3, 3], [1, 1], [1, 1]), kwargs = {})
#   %add_33 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_18, %avg_pool2d_1), kwargs = {})
triton_poi_fused_add_avg_pool2d_11 = async_compile.triton('triton_poi_fused_add_avg_pool2d_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + x3), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-16) + x3), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-15) + x3), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x3), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x3), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (15 + x3), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (16 + x3), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (17 + x3), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17)))*((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17))) + ((-1)*x0*((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17)))) + ((-1)*x1*((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17)))) + ((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17))) + ((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3spnsmaj6bkqqney66tgnodouwwrdmtvr6ak7hvjivwznojy7w.py
# Topologically Sorted Source Nodes: [output_8, output_11, batch_norm_8, output_14, output2, batch_norm_9, output_16, long2, output2_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   batch_norm_8 => add_42, mul_40, mul_41, sub_8
#   batch_norm_9 => add_45, mul_44, mul_45, sub_9
#   long2 => add_46
#   output2 => add_43
#   output2_add_1 => add_47
#   output_11 => gt_6, mul_34, where_6
#   output_14 => gt_7, mul_42, where_7
#   output_16 => gt_8, mul_46, where_8
#   output_8 => gt_5, mul_26, where_5
# Graph fragment:
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_20, %mul_26), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %add_31), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_31, %mul_34), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_73), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_75), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_77), kwargs = {})
#   %add_42 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_79), kwargs = {})
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_42, 0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %add_42), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_42, %mul_42), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_7, %where_6), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_81), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_83), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_85), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_87), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_45, 0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %add_45), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_45, %mul_46), kwargs = {})
#   %add_46 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %where_5), kwargs = {})
#   %add_47 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %add_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 24)
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
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x3), None)
    tmp46 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp34 > tmp29
    tmp37 = tmp36 * tmp34
    tmp38 = tl.where(tmp35, tmp34, tmp37)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp28 > tmp29
    tmp42 = tmp41 * tmp28
    tmp43 = tl.where(tmp40, tmp28, tmp42)
    tmp45 = tmp44 > tmp29
    tmp47 = tmp46 * tmp44
    tmp48 = tl.where(tmp45, tmp44, tmp47)
    tmp49 = tmp43 + tmp48
    tmp50 = tmp39 + tmp49
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: inductor_cache/uo/cuoqsypdigzsv7fpqjtripkaipa5uz32a5oit3rwpvxicavockzl.py
# Topologically Sorted Source Nodes: [output_8, output_11, output_14, output2, output_16, long2, batch_norm_10, output_19, output2_1, output2_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_10 => add_57, mul_52, mul_53, sub_10
#   long2 => add_46
#   output2 => add_43
#   output2_1 => add_58
#   output2_add_2 => add_59
#   output_11 => gt_6, mul_34, where_6
#   output_14 => gt_7, mul_42, where_7
#   output_16 => gt_8, mul_46, where_8
#   output_19 => gt_9, mul_54, where_9
#   output_8 => gt_5, mul_26, where_5
# Graph fragment:
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_20, %mul_26), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %add_31), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_31, %mul_34), kwargs = {})
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_42, 0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %add_42), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_42, %mul_42), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_7, %where_6), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_45, 0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %add_45), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_45, %mul_46), kwargs = {})
#   %add_46 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %where_5), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_93), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_95), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_97), kwargs = {})
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_99), kwargs = {})
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_57, 0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %add_57), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_57, %mul_54), kwargs = {})
#   %add_58 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_9, %add_43), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %add_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 24)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
    tmp28 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x3), None)
    tmp35 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr12 + (x3), None)
    tmp40 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp26 > tmp16
    tmp29 = tmp28 * tmp26
    tmp30 = tl.where(tmp27, tmp26, tmp29)
    tmp31 = tmp25 + tmp30
    tmp32 = tmp20 + tmp31
    tmp34 = tmp33 > tmp16
    tmp36 = tmp35 * tmp33
    tmp37 = tl.where(tmp34, tmp33, tmp36)
    tmp39 = tmp38 > tmp16
    tmp41 = tmp40 * tmp38
    tmp42 = tl.where(tmp39, tmp38, tmp41)
    tmp43 = tmp37 + tmp42
    tmp44 = tmp32 + tmp43
    tl.store(in_out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr0 + (x3), tmp44, None)
''', device_str='cuda')


# kernel path: inductor_cache/oc/coccsqq624mhq3fq43fbvic4wq7o7hxlstayar2anbpcdkgoz5uv.py
# Topologically Sorted Source Nodes: [batch_norm_11, output_22, output2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_11 => add_69, mul_60, mul_61, sub_11
#   output2_2 => add_70
#   output_22 => gt_10, mul_62, where_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_105), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_107), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %unsqueeze_109), kwargs = {})
#   %add_69 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %unsqueeze_111), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_69, 0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_10, %add_69), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %add_69, %mul_62), kwargs = {})
#   %add_70 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_10, %add_58), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 24)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6kjnbusofodht7zy2rc6spyc7wlitdwfav46smncebbcmbropjg.py
# Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_9 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_46, %add_70], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 48)
    x0 = (xindex % 256)
    x2 = xindex // 12288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 6144*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 * tmp5
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.load(in_ptr2 + (x0 + 256*(x1) + 6144*x2), tmp4, other=0.0)
    tmp12 = tmp11 > tmp6
    tmp13 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp11
    tmp15 = tl.where(tmp12, tmp11, tmp14)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 48, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + (x0 + 256*((-24) + x1) + 6144*x2), tmp19, other=0.0)
    tmp23 = tl.where(tmp4, tmp18, tmp22)
    tl.store(out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/tc/ctcrbvicc423ifp4ytizpp5ymd5jrzsy3h6vmrjvt77et5j6g2fp.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_4 => add_73, mul_64, mul_65, sub_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_113), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_115), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_117), kwargs = {})
#   %add_73 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 48)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7a/c7a3qpxchnxcyfdvkv4cvozhekchzewoazl4d3xhrcvh2poogzto.py
# Topologically Sorted Source Nodes: [output_8, output_16, long2, x1_1, add_48], Original ATen: [aten._prelu_kernel, aten.add, aten.clone]
# Source node to ATen node mapping:
#   add_48 => add_74
#   long2 => add_46
#   output_16 => gt_8, mul_46, where_8
#   output_8 => gt_5, mul_26, where_5
#   x1_1 => clone_2
# Graph fragment:
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_20, %mul_26), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_45, 0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %add_45), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_45, %mul_46), kwargs = {})
#   %add_46 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %where_5), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_2, %add_46), kwargs = {})
triton_poi_fused__prelu_kernel_add_clone_17 = async_compile.triton('triton_poi_fused__prelu_kernel_add_clone_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_clone_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_clone_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 6144
    x3 = (xindex % 6144)
    x4 = xindex
    x1 = ((xindex // 256) % 24)
    tmp0 = tl.load(in_ptr0 + (x3 + 12288*x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tmp7 > tmp2
    tmp10 = tmp9 * tmp7
    tmp11 = tl.where(tmp8, tmp7, tmp10)
    tmp12 = tmp6 + tmp11
    tmp13 = tmp0 + tmp12
    tl.store(in_out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/i5/ci5xrqqihjgcouqqgbrq364ng3dkzzno2jv27bpetoz467ijymed.py
# Topologically Sorted Source Nodes: [x2_1, add_49], Original ATen: [aten.clone, aten.add]
# Source node to ATen node mapping:
#   add_49 => add_77
#   x2_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_30,), kwargs = {memory_format: torch.contiguous_format})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_3, %add_70), kwargs = {})
triton_poi_fused_add_clone_18 = async_compile.triton('triton_poi_fused_add_clone_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (6144 + x0 + 12288*x1), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/fb/cfbf2moda7344i762lliqrsbr4kiktaat5rvduknn6fedzxklqa4.py
# Topologically Sorted Source Nodes: [p_4, d1_13], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_13 => add_78
#   p_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_43, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %add_78 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_44, %avg_pool2d_4), kwargs = {})
triton_poi_fused_add_avg_pool2d_19 = async_compile.triton('triton_poi_fused_add_avg_pool2d_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x4 = xindex // 8
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + 2*x0 + 32*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-16) + 2*x0 + 32*x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-15) + 2*x0 + 32*x4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 32*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 32*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (15 + 2*x0 + 32*x4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17)))*((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17))) + ((-2)*x0*((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17)))) + ((-2)*x1*((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17)))) + 4*x0*x1 + ((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17))) + ((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bh/cbhs76lk4rqeqomrbq7t4svhkxrcb3s3lwuqhnn6dj3lgn3hejbl.py
# Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_10 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_78, %add_79, %add_80, %add_81], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 32)
    x0 = (xindex % 64)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 512*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 64*((-8) + x1) + 512*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + 64*((-8) + x1) + 512*x2), tmp9, other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp0 >= tmp7
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + (x0 + 64*((-16) + x1) + 512*x2), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr1 + (x0 + 64*((-16) + x1) + 512*x2), tmp18, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr2 + (x0 + 64*((-16) + x1) + 512*x2), tmp18, other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = tmp0 >= tmp16
    tmp27 = tl.full([1], 32, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 64*((-24) + x1) + 512*x2), tmp26, other=0.0)
    tmp30 = tl.load(in_ptr1 + (x0 + 64*((-24) + x1) + 512*x2), tmp26, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr2 + (x0 + 64*((-24) + x1) + 512*x2), tmp26, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr3 + (x0 + 64*((-24) + x1) + 512*x2), tmp26, other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp18, tmp25, tmp37)
    tmp39 = tl.where(tmp9, tmp14, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x3), tmp40, None)
''', device_str='cuda')


# kernel path: inductor_cache/yo/cyoi7jsfbnea4t23cido5rmv3m5uzl7lti2auvdhdf3jvit4q66d.py
# Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_11 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_82, %add_83, %add_84, %add_85], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 32)
    x0 = (xindex % 64)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 512*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 16, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (x0 + 64*((-8) + x1) + 512*x2), tmp15, other=0.0)
    tmp17 = tl.load(in_ptr2 + (x0 + 64*((-8) + x1) + 512*x2), tmp15, other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.load(in_ptr1 + (64 + x0 + 256*x2), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tmp0 >= tmp13
    tmp26 = tl.full([1], 24, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr3 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr1 + (128 + x0 + 256*x2), tmp28, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp28, tmp37, tmp38)
    tmp40 = tmp0 >= tmp26
    tmp41 = tl.full([1], 32, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tl.load(in_ptr0 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp44 = tl.load(in_ptr2 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr3 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr4 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.load(in_ptr1 + (192 + x0 + 256*x2), tmp40, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp49 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp40, tmp53, tmp54)
    tmp56 = tl.where(tmp28, tmp39, tmp55)
    tmp57 = tl.where(tmp15, tmp24, tmp56)
    tmp58 = tl.where(tmp4, tmp11, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/sr/csrrwzjbzso3zxhotech4ntfcssfellhdzlbysunljl354hv4fk6.py
# Topologically Sorted Source Nodes: [batch_norm_13, output_24, batch_norm_14, output_27, output3_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_13 => add_76, mul_67, mul_68, sub_13
#   batch_norm_14 => add_87, mul_75, mul_76, sub_14
#   output3_add => add_88
#   output_24 => gt_11, mul_69, where_11
#   output_27 => gt_12, mul_77, where_12
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_121), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_123), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_125), kwargs = {})
#   %add_76 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_127), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_76, 0), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %add_76), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_76, %mul_69), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_133), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_135), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %unsqueeze_137), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, %unsqueeze_139), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_87, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_12, %add_87), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_87, %mul_77), kwargs = {})
#   %add_88 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_12, %where_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp28 > tmp29
    tmp32 = tmp31 * tmp28
    tmp33 = tl.where(tmp30, tmp28, tmp32)
    tmp34 = tmp15 > tmp29
    tmp36 = tmp35 * tmp15
    tmp37 = tl.where(tmp34, tmp15, tmp36)
    tmp38 = tmp33 + tmp37
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7ghounau6y6cmwspibvb7ferbfjonolr275u5d4j2bgo52lvqv2.py
# Topologically Sorted Source Nodes: [p_5, d1_16], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_16 => add_89
#   p_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_50, [3, 3], [1, 1], [1, 1]), kwargs = {})
#   %add_89 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_51, %avg_pool2d_5), kwargs = {})
triton_poi_fused_add_avg_pool2d_23 = async_compile.triton('triton_poi_fused_add_avg_pool2d_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9) + x3), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-8) + x3), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7) + x3), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x3), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x3), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7 + x3), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (8 + x3), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9 + x3), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((9) * ((9) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (9)))*((9) * ((9) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (9))) + ((-1)*x0*((9) * ((9) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (9)))) + ((-1)*x1*((9) * ((9) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (9)))) + ((9) * ((9) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (9))) + ((9) * ((9) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (9)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ta/ctac3zf5u3ebe7oaxgnebrvt4tsymshfqvrfwi2g7qdqqy3ayhqv.py
# Topologically Sorted Source Nodes: [output_24, output_27, batch_norm_15, output_30, output3, batch_norm_16, output_32, long3, output3_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   batch_norm_15 => add_98, mul_83, mul_84, sub_15
#   batch_norm_16 => add_101, mul_87, mul_88, sub_16
#   long3 => add_102
#   output3 => add_99
#   output3_add_1 => add_103
#   output_24 => gt_11, mul_69, where_11
#   output_27 => gt_12, mul_77, where_12
#   output_30 => gt_13, mul_85, where_13
#   output_32 => gt_14, mul_89, where_14
# Graph fragment:
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_76, 0), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %add_76), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_76, %mul_69), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_87, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_12, %add_87), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_87, %mul_77), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_145), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_147), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %unsqueeze_149), kwargs = {})
#   %add_98 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_151), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_98, 0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %add_98), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %add_98, %mul_85), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_13, %where_12), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_153), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_155), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_157), kwargs = {})
#   %add_101 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_159), kwargs = {})
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_101, 0), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %add_101), kwargs = {})
#   %where_14 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_101, %mul_89), kwargs = {})
#   %add_102 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_14, %where_11), kwargs = {})
#   %add_103 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %add_102), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x3), None)
    tmp46 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp34 > tmp29
    tmp37 = tmp36 * tmp34
    tmp38 = tl.where(tmp35, tmp34, tmp37)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp28 > tmp29
    tmp42 = tmp41 * tmp28
    tmp43 = tl.where(tmp40, tmp28, tmp42)
    tmp45 = tmp44 > tmp29
    tmp47 = tmp46 * tmp44
    tmp48 = tl.where(tmp45, tmp44, tmp47)
    tmp49 = tmp43 + tmp48
    tmp50 = tmp39 + tmp49
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: inductor_cache/qh/cqheyon66dgs3rtdigkofaf2nawmukgackcrvwu3jtmjshfdddth.py
# Topologically Sorted Source Nodes: [output_24, output_27, output_30, output3, output_32, long3, batch_norm_17, output_35, output3_1, batch_norm_18, output_37, long3_1, output3_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_17 => add_113, mul_95, mul_96, sub_17
#   batch_norm_18 => add_116, mul_100, mul_99, sub_18
#   long3 => add_102
#   long3_1 => add_117
#   output3 => add_99
#   output3_1 => add_114
#   output3_add_2 => add_118
#   output_24 => gt_11, mul_69, where_11
#   output_27 => gt_12, mul_77, where_12
#   output_30 => gt_13, mul_85, where_13
#   output_32 => gt_14, mul_89, where_14
#   output_35 => gt_15, mul_97, where_15
#   output_37 => gt_16, mul_101, where_16
# Graph fragment:
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_76, 0), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %add_76), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_76, %mul_69), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_87, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_12, %add_87), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_87, %mul_77), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_98, 0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %add_98), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %add_98, %mul_85), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_13, %where_12), kwargs = {})
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_101, 0), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %add_101), kwargs = {})
#   %where_14 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_101, %mul_89), kwargs = {})
#   %add_102 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_14, %where_11), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_165), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_167), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %unsqueeze_169), kwargs = {})
#   %add_113 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %unsqueeze_171), kwargs = {})
#   %gt_15 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_113, 0), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %add_113), kwargs = {})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_15, %add_113, %mul_97), kwargs = {})
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_15, %add_99), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_173), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_175), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_99, %unsqueeze_177), kwargs = {})
#   %add_116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_100, %unsqueeze_179), kwargs = {})
#   %gt_16 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_116, 0), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %add_116), kwargs = {})
#   %where_16 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_16, %add_116, %mul_101), kwargs = {})
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_16, %add_102), kwargs = {})
#   %add_118 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_114, %add_117), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_25(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
    tmp28 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x3), None)
    tmp34 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr16 + (x3), None)
    tmp52 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr18 + (x3), None)
    tmp57 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp26 > tmp16
    tmp29 = tmp28 * tmp26
    tmp30 = tl.where(tmp27, tmp26, tmp29)
    tmp31 = tmp25 + tmp30
    tmp32 = tmp20 + tmp31
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = tmp7 / tmp38
    tmp40 = tmp39 * tmp9
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 > tmp16
    tmp48 = tmp47 * tmp45
    tmp49 = tl.where(tmp46, tmp45, tmp48)
    tmp51 = tmp50 > tmp16
    tmp53 = tmp52 * tmp50
    tmp54 = tl.where(tmp51, tmp50, tmp53)
    tmp56 = tmp55 > tmp16
    tmp58 = tmp57 * tmp55
    tmp59 = tl.where(tmp56, tmp55, tmp58)
    tmp60 = tmp54 + tmp59
    tmp61 = tmp49 + tmp60
    tmp62 = tmp32 + tmp61
    tl.store(in_out_ptr0 + (x3), tmp32, None)
    tl.store(in_out_ptr1 + (x3), tmp61, None)
    tl.store(out_ptr0 + (x3), tmp62, None)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwvp6ga6xttke3jc7uvozflfviyzkzomaemlhcaoyrszagx3jsx.py
# Topologically Sorted Source Nodes: [batch_norm_19, output_40, output3_2, batch_norm_20, output_42, long3_2, output3_add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_19 => add_128, mul_107, mul_108, sub_19
#   batch_norm_20 => add_131, mul_111, mul_112, sub_20
#   long3_2 => add_132
#   output3_2 => add_129
#   output3_add_3 => add_133
#   output_40 => gt_17, mul_109, where_17
#   output_42 => gt_18, mul_113, where_18
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_74, %unsqueeze_185), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_187), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %unsqueeze_189), kwargs = {})
#   %add_128 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %unsqueeze_191), kwargs = {})
#   %gt_17 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_128, 0), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %add_128), kwargs = {})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_17, %add_128, %mul_109), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_17, %add_114), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_193), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_195), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, %unsqueeze_197), kwargs = {})
#   %add_131 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %unsqueeze_199), kwargs = {})
#   %gt_18 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_131, 0), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, %add_131), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %add_131, %mul_113), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %add_117), kwargs = {})
#   %add_133 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_129, %add_132), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp37 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x3), None)
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp28 > tmp29
    tmp38 = tmp37 * tmp28
    tmp39 = tl.where(tmp36, tmp28, tmp38)
    tmp41 = tmp39 + tmp40
    tmp42 = tmp35 + tmp41
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwa3eyyg5yor2gajvn5wr535qr6ez3aknslpngbmerfubro37ueb.py
# Topologically Sorted Source Nodes: [output_40, output3_2, output_42, long3_2, batch_norm_21, output_45, output3_3, batch_norm_22, output_47, long3_3, output3_add_4], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_21 => add_143, mul_119, mul_120, sub_21
#   batch_norm_22 => add_146, mul_123, mul_124, sub_22
#   long3_2 => add_132
#   long3_3 => add_147
#   output3_2 => add_129
#   output3_3 => add_144
#   output3_add_4 => add_148
#   output_40 => gt_17, mul_109, where_17
#   output_42 => gt_18, mul_113, where_18
#   output_45 => gt_19, mul_121, where_19
#   output_47 => gt_20, mul_125, where_20
# Graph fragment:
#   %gt_17 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_128, 0), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %add_128), kwargs = {})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_17, %add_128, %mul_109), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_17, %add_114), kwargs = {})
#   %gt_18 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_131, 0), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, %add_131), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %add_131, %mul_113), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %add_117), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_205), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_207), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_209), kwargs = {})
#   %add_143 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_211), kwargs = {})
#   %gt_19 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_143, 0), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %add_143), kwargs = {})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %add_143, %mul_121), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_19, %add_129), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_213), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_215), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_123, %unsqueeze_217), kwargs = {})
#   %add_146 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %unsqueeze_219), kwargs = {})
#   %gt_20 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_146, 0), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %add_146), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %add_146, %mul_125), kwargs = {})
#   %add_147 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_20, %add_132), kwargs = {})
#   %add_148 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %add_147), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 64) % 32)
    x2 = xindex // 2048
    x3 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x4), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x4), None)
    tmp29 = tl.load(in_ptr9 + (x4), None)
    tmp30 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr15 + (x4), None)
    tmp48 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr17 + (x4), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp20 + tmp27
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp4
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp7 / tmp34
    tmp36 = tmp35 * tmp9
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp41 > tmp16
    tmp44 = tmp43 * tmp41
    tmp45 = tl.where(tmp42, tmp41, tmp44)
    tmp47 = tmp46 > tmp16
    tmp49 = tmp48 * tmp46
    tmp50 = tl.where(tmp47, tmp46, tmp49)
    tmp52 = tmp50 + tmp51
    tmp53 = tmp45 + tmp52
    tmp54 = tmp28 + tmp53
    tl.store(in_out_ptr0 + (x4), tmp28, None)
    tl.store(out_ptr1 + (x3 + 4096*x2), tmp53, None)
    tl.store(out_ptr2 + (x4), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/pr/cpr4bwoinoabaaxf5l4xy3ryvopyewktq63ofea4hs5z7jdhqswy.py
# Topologically Sorted Source Nodes: [batch_norm_23, output_50, output3_4, output3_add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_23 => add_158, mul_131, mul_132, sub_23
#   output3_4 => add_159
#   output3_add_5 => add_160
#   output_50 => gt_21, mul_133, where_21
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_225), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_227), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %unsqueeze_229), kwargs = {})
#   %add_158 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_132, %unsqueeze_231), kwargs = {})
#   %gt_21 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_158, 0), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %add_158), kwargs = {})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_21, %add_158, %mul_133), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_21, %add_144), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %add_147), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 32)
    x2 = xindex // 2048
    x4 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x4 + 4096*x2), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/a4/ca4s7lwtnhziklcbiygpsl3vraheufhxpa5rbc2yrm7j6irieulr.py
# Topologically Sorted Source Nodes: [output_50, output3_4, batch_norm_24, output_53, output3_5, output3_add_6], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_24 => add_170, mul_139, mul_140, sub_24
#   output3_4 => add_159
#   output3_5 => add_171
#   output3_add_6 => add_172
#   output_50 => gt_21, mul_133, where_21
#   output_53 => gt_22, mul_141, where_22
# Graph fragment:
#   %gt_21 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_158, 0), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %add_158), kwargs = {})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_21, %add_158, %mul_133), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_21, %add_144), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_237), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_239), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_241), kwargs = {})
#   %add_170 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_243), kwargs = {})
#   %gt_22 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_170, 0), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_170), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %add_170, %mul_141), kwargs = {})
#   %add_171 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_22, %add_159), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_171, %add_147), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 64) % 32)
    x2 = xindex // 2048
    x3 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x4), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x4), None)
    tmp29 = tl.load(in_ptr9 + (x3 + 4096*x2), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp20 + tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x4), tmp28, None)
    tl.store(out_ptr0 + (x4), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/dk/cdkitnnnjv353raqr5hdblholxsacu6pifwppb2kgnandwppek5f.py
# Topologically Sorted Source Nodes: [output_56, output3_6, batch_norm_26, output_59, output3_7, cat_28], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_26 => add_194, mul_155, mul_156, sub_26
#   cat_28 => cat_28
#   output3_6 => add_183
#   output3_7 => add_195
#   output_56 => gt_23, mul_149, where_23
#   output_59 => gt_24, mul_157, where_24
# Graph fragment:
#   %gt_23 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_182, 0), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_23, %add_182), kwargs = {})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %add_182, %mul_149), kwargs = {})
#   %add_183 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_23, %add_171), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_261), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_263), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %unsqueeze_265), kwargs = {})
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %unsqueeze_267), kwargs = {})
#   %gt_24 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_194, 0), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, %add_194), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %add_194, %mul_157), kwargs = {})
#   %add_195 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_24, %add_183), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_147, %add_195], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_cat_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_cat_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_cat_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_cat_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 64) % 32)
    x2 = xindex // 2048
    x3 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x4), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x4), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp20 + tmp27
    tl.store(in_out_ptr0 + (x4), tmp28, None)
    tl.store(out_ptr0 + (x3 + 4096*x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvtjnsonfs6qu4inoowm67lnivaa3ohstz5efv36ghyuzztcwbz.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_6 => add_198, mul_159, mul_160, sub_27
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_269), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_271), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_273), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_275), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/qi/cqif4kb7psiotquso5kqxllyve5ghh7pstvrgax5nr3kwxsbuvfb.py
# Topologically Sorted Source Nodes: [x1_2, add_143], Original ATen: [aten.clone, aten.add]
# Source node to ATen node mapping:
#   add_143 => add_199
#   x1_2 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_70,), kwargs = {memory_format: torch.contiguous_format})
#   %add_199 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_4, %add_147), kwargs = {})
triton_poi_fused_add_clone_32 = async_compile.triton('triton_poi_fused_add_clone_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_32(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 4096*x1), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjwb5xbldfnu32rxr3yy2enk7ydaon2fmo72dd3gdgqk3p5kxd6.py
# Topologically Sorted Source Nodes: [x2_2, add_144], Original ATen: [aten.clone, aten.add]
# Source node to ATen node mapping:
#   add_144 => add_202
#   x2_2 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_74,), kwargs = {memory_format: torch.contiguous_format})
#   %add_202 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_5, %add_195), kwargs = {})
triton_poi_fused_add_clone_33 = async_compile.triton('triton_poi_fused_add_clone_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_33(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 4096*x1), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/kk/ckkqqa3amlotvo6zgqcccoydkjb2v5ccgbzrblhgpss3ol3g76w6.py
# Topologically Sorted Source Nodes: [p_13, d1_40], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_40 => add_203
#   p_13 => avg_pool2d_13
# Graph fragment:
#   %avg_pool2d_13 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_117, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %add_203 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_118, %avg_pool2d_13), kwargs = {})
triton_poi_fused_add_avg_pool2d_34 = async_compile.triton('triton_poi_fused_add_avg_pool2d_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex // 4
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9) + 2*x0 + 16*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-8) + 2*x0 + 16*x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7) + 2*x0 + 16*x4), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 16*x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 16*x4), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x4), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7 + 2*x0 + 16*x4), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x4), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x4), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9)))*((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9))) + ((-2)*x0*((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9)))) + ((-2)*x1*((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9)))) + 4*x0*x1 + ((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9))) + ((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r3/cr3jlqqtiji4pqn7qsg36x6xwjedv3oxautr6utae4odvzlwmqwo.py
# Topologically Sorted Source Nodes: [cat_29], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_29 => cat_29
# Graph fragment:
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_203, %add_204, %add_205, %add_206], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 64)
    x0 = (xindex % 16)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 256*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 16*((-16) + x1) + 256*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + 16*((-16) + x1) + 256*x2), tmp9, other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp0 >= tmp7
    tmp16 = tl.full([1], 48, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + (x0 + 16*((-32) + x1) + 256*x2), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr1 + (x0 + 16*((-32) + x1) + 256*x2), tmp18, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr2 + (x0 + 16*((-32) + x1) + 256*x2), tmp18, other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = tmp0 >= tmp16
    tmp27 = tl.full([1], 64, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 16*((-48) + x1) + 256*x2), tmp26, other=0.0)
    tmp30 = tl.load(in_ptr1 + (x0 + 16*((-48) + x1) + 256*x2), tmp26, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr2 + (x0 + 16*((-48) + x1) + 256*x2), tmp26, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr3 + (x0 + 16*((-48) + x1) + 256*x2), tmp26, other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp18, tmp25, tmp37)
    tmp39 = tl.where(tmp9, tmp14, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x3), tmp40, None)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvku4udfn4nsdvaftvuqltt6rlfuvouo5rtrjt6iyndfvgbp2fo.py
# Topologically Sorted Source Nodes: [cat_30], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_30 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_207, %add_208, %add_209, %add_210], 1), kwargs = {})
triton_poi_fused_cat_36 = async_compile.triton('triton_poi_fused_cat_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 64)
    x0 = (xindex % 16)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 256*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 32, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (x0 + 16*((-16) + x1) + 256*x2), tmp15, other=0.0)
    tmp17 = tl.load(in_ptr2 + (x0 + 16*((-16) + x1) + 256*x2), tmp15, other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.load(in_ptr1 + (16 + x0 + 64*x2), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tmp0 >= tmp13
    tmp26 = tl.full([1], 48, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 16*((-32) + x1) + 256*x2), tmp28, other=0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + 16*((-32) + x1) + 256*x2), tmp28, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr3 + (x0 + 16*((-32) + x1) + 256*x2), tmp28, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr1 + (32 + x0 + 64*x2), tmp28, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp28, tmp37, tmp38)
    tmp40 = tmp0 >= tmp26
    tmp41 = tl.full([1], 64, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tl.load(in_ptr0 + (x0 + 16*((-48) + x1) + 256*x2), tmp40, other=0.0)
    tmp44 = tl.load(in_ptr2 + (x0 + 16*((-48) + x1) + 256*x2), tmp40, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr3 + (x0 + 16*((-48) + x1) + 256*x2), tmp40, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr4 + (x0 + 16*((-48) + x1) + 256*x2), tmp40, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.load(in_ptr1 + (48 + x0 + 64*x2), tmp40, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp49 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp40, tmp53, tmp54)
    tmp56 = tl.where(tmp28, tmp39, tmp55)
    tmp57 = tl.where(tmp15, tmp24, tmp56)
    tmp58 = tl.where(tmp4, tmp11, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqxrffyta3s2pfe7cjq3efnynwmlownr4qee73afjzexe5jfd6j6.py
# Topologically Sorted Source Nodes: [batch_norm_28, output_61, batch_norm_29, output_64, output4_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_28 => add_201, mul_162, mul_163, sub_28
#   batch_norm_29 => add_212, mul_170, mul_171, sub_29
#   output4_add => add_213
#   output_61 => gt_25, mul_164, where_25
#   output_64 => gt_26, mul_172, where_26
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_277), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_279), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_281), kwargs = {})
#   %add_201 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_283), kwargs = {})
#   %gt_25 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_201, 0), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %add_201), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %add_201, %mul_164), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_289), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_291), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_170, %unsqueeze_293), kwargs = {})
#   %add_212 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_171, %unsqueeze_295), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_212, 0), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %add_212), kwargs = {})
#   %where_26 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %add_212, %mul_172), kwargs = {})
#   %add_213 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_26, %where_25), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp28 > tmp29
    tmp32 = tmp31 * tmp28
    tmp33 = tl.where(tmp30, tmp28, tmp32)
    tmp34 = tmp15 > tmp29
    tmp36 = tmp35 * tmp15
    tmp37 = tl.where(tmp34, tmp15, tmp36)
    tmp38 = tmp33 + tmp37
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/cco3jkkzyly5xmbaxqomx2eogy5b2dcippznxrpi4xwzsh75anxo.py
# Topologically Sorted Source Nodes: [p_14, d1_43], Original ATen: [aten.avg_pool2d, aten.add]
# Source node to ATen node mapping:
#   d1_43 => add_214
#   p_14 => avg_pool2d_14
# Graph fragment:
#   %avg_pool2d_14 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_124, [3, 3], [1, 1], [1, 1]), kwargs = {})
#   %add_214 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_125, %avg_pool2d_14), kwargs = {})
triton_poi_fused_add_avg_pool2d_38 = async_compile.triton('triton_poi_fused_add_avg_pool2d_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x3 = xindex
    tmp54 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + x3), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + x3), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + x3), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x3), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x3), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + x3), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + x3), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + x3), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5)))*((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5))) + ((-1)*x0*((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5)))) + ((-1)*x1*((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5)))) + ((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5))) + ((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5)))
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 + tmp53
    tl.store(in_out_ptr0 + (x3), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b2/cb22q42a3atksxz5fk3t323ds2nkqnljwzkdpuhw6iazsuns3gbh.py
# Topologically Sorted Source Nodes: [output_61, output_64, batch_norm_30, output_67, output4, batch_norm_31, output_69, long4, output4_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   batch_norm_30 => add_223, mul_178, mul_179, sub_30
#   batch_norm_31 => add_226, mul_182, mul_183, sub_31
#   long4 => add_227
#   output4 => add_224
#   output4_add_1 => add_228
#   output_61 => gt_25, mul_164, where_25
#   output_64 => gt_26, mul_172, where_26
#   output_67 => gt_27, mul_180, where_27
#   output_69 => gt_28, mul_184, where_28
# Graph fragment:
#   %gt_25 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_201, 0), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %add_201), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %add_201, %mul_164), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_212, 0), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %add_212), kwargs = {})
#   %where_26 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %add_212, %mul_172), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_301), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_303), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_305), kwargs = {})
#   %add_223 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_307), kwargs = {})
#   %gt_27 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_223, 0), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %add_223), kwargs = {})
#   %where_27 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %add_223, %mul_180), kwargs = {})
#   %add_224 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_27, %where_26), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_309), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_311), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_313), kwargs = {})
#   %add_226 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %unsqueeze_315), kwargs = {})
#   %gt_28 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_226, 0), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_226), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %add_226, %mul_184), kwargs = {})
#   %add_227 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_28, %where_25), kwargs = {})
#   %add_228 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_224, %add_227), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x3), None)
    tmp46 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp34 > tmp29
    tmp37 = tmp36 * tmp34
    tmp38 = tl.where(tmp35, tmp34, tmp37)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp28 > tmp29
    tmp42 = tmp41 * tmp28
    tmp43 = tl.where(tmp40, tmp28, tmp42)
    tmp45 = tmp44 > tmp29
    tmp47 = tmp46 * tmp44
    tmp48 = tl.where(tmp45, tmp44, tmp47)
    tmp49 = tmp43 + tmp48
    tmp50 = tmp39 + tmp49
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: inductor_cache/jg/cjg2ytl2edxdb3ynfniagqj6vnues4hcquws6fkdyuawudotbjba.py
# Topologically Sorted Source Nodes: [output_61, output_64, output_67, output4, output_69, long4, batch_norm_32, output_72, output4_1, batch_norm_33, output_74, long4_1, output4_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_32 => add_238, mul_190, mul_191, sub_32
#   batch_norm_33 => add_241, mul_194, mul_195, sub_33
#   long4 => add_227
#   long4_1 => add_242
#   output4 => add_224
#   output4_1 => add_239
#   output4_add_2 => add_243
#   output_61 => gt_25, mul_164, where_25
#   output_64 => gt_26, mul_172, where_26
#   output_67 => gt_27, mul_180, where_27
#   output_69 => gt_28, mul_184, where_28
#   output_72 => gt_29, mul_192, where_29
#   output_74 => gt_30, mul_196, where_30
# Graph fragment:
#   %gt_25 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_201, 0), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %add_201), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %add_201, %mul_164), kwargs = {})
#   %gt_26 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_212, 0), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %add_212), kwargs = {})
#   %where_26 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_26, %add_212, %mul_172), kwargs = {})
#   %gt_27 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_223, 0), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %add_223), kwargs = {})
#   %where_27 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %add_223, %mul_180), kwargs = {})
#   %add_224 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_27, %where_26), kwargs = {})
#   %gt_28 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_226, 0), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_226), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %add_226, %mul_184), kwargs = {})
#   %add_227 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_28, %where_25), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_139, %unsqueeze_321), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_323), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_325), kwargs = {})
#   %add_238 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_327), kwargs = {})
#   %gt_29 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_238, 0), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %add_238), kwargs = {})
#   %where_29 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %add_238, %mul_192), kwargs = {})
#   %add_239 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_29, %add_224), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_329), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_331), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_194, %unsqueeze_333), kwargs = {})
#   %add_241 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %unsqueeze_335), kwargs = {})
#   %gt_30 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_241, 0), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_30, %add_241), kwargs = {})
#   %where_30 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %add_241, %mul_196), kwargs = {})
#   %add_242 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_30, %add_227), kwargs = {})
#   %add_243 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_239, %add_242), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_40(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
    tmp28 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x3), None)
    tmp34 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr16 + (x3), None)
    tmp52 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr18 + (x3), None)
    tmp57 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp26 > tmp16
    tmp29 = tmp28 * tmp26
    tmp30 = tl.where(tmp27, tmp26, tmp29)
    tmp31 = tmp25 + tmp30
    tmp32 = tmp20 + tmp31
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = tmp7 / tmp38
    tmp40 = tmp39 * tmp9
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 > tmp16
    tmp48 = tmp47 * tmp45
    tmp49 = tl.where(tmp46, tmp45, tmp48)
    tmp51 = tmp50 > tmp16
    tmp53 = tmp52 * tmp50
    tmp54 = tl.where(tmp51, tmp50, tmp53)
    tmp56 = tmp55 > tmp16
    tmp58 = tmp57 * tmp55
    tmp59 = tl.where(tmp56, tmp55, tmp58)
    tmp60 = tmp54 + tmp59
    tmp61 = tmp49 + tmp60
    tmp62 = tmp32 + tmp61
    tl.store(in_out_ptr0 + (x3), tmp32, None)
    tl.store(in_out_ptr1 + (x3), tmp61, None)
    tl.store(out_ptr0 + (x3), tmp62, None)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxg7yiaft6fj2tktxpi2v5ih4aw6bunfeegmoi3lnbtug46dtfsf.py
# Topologically Sorted Source Nodes: [batch_norm_34, output_77, output4_2, batch_norm_35, output_79, long4_2, output4_add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   batch_norm_34 => add_253, mul_202, mul_203, sub_34
#   batch_norm_35 => add_256, mul_206, mul_207, sub_35
#   long4_2 => add_257
#   output4_2 => add_254
#   output4_add_3 => add_258
#   output_77 => gt_31, mul_204, where_31
#   output_79 => gt_32, mul_208, where_32
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_148, %unsqueeze_341), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_343), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_345), kwargs = {})
#   %add_253 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_347), kwargs = {})
#   %gt_31 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_253, 0), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_31, %add_253), kwargs = {})
#   %where_31 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_31, %add_253, %mul_204), kwargs = {})
#   %add_254 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_31, %add_239), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_150, %unsqueeze_349), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_351), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %unsqueeze_353), kwargs = {})
#   %add_256 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %unsqueeze_355), kwargs = {})
#   %gt_32 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_256, 0), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_32, %add_256), kwargs = {})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %add_256, %mul_208), kwargs = {})
#   %add_257 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_32, %add_242), kwargs = {})
#   %add_258 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_254, %add_257), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
    tmp37 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x3), None)
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
    tmp29 = 0.0
    tmp30 = tmp15 > tmp29
    tmp32 = tmp31 * tmp15
    tmp33 = tl.where(tmp30, tmp15, tmp32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp28 > tmp29
    tmp38 = tmp37 * tmp28
    tmp39 = tl.where(tmp36, tmp28, tmp38)
    tmp41 = tmp39 + tmp40
    tmp42 = tmp35 + tmp41
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/cco4dkkh3p75bnp254axej5yppfdsnykyxc6lh7hpthsl2nftu6t.py
# Topologically Sorted Source Nodes: [output_77, output4_2, output_79, long4_2, batch_norm_36, output_82, output4_3, output4_add_4], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_36 => add_268, mul_214, mul_215, sub_36
#   long4_2 => add_257
#   output4_2 => add_254
#   output4_3 => add_269
#   output4_add_4 => add_270
#   output_77 => gt_31, mul_204, where_31
#   output_79 => gt_32, mul_208, where_32
#   output_82 => gt_33, mul_216, where_33
# Graph fragment:
#   %gt_31 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_253, 0), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_31, %add_253), kwargs = {})
#   %where_31 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_31, %add_253, %mul_204), kwargs = {})
#   %add_254 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_31, %add_239), kwargs = {})
#   %gt_32 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_256, 0), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_32, %add_256), kwargs = {})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %add_256, %mul_208), kwargs = {})
#   %add_257 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_32, %add_242), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_157, %unsqueeze_361), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_363), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_365), kwargs = {})
#   %add_268 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_367), kwargs = {})
#   %gt_33 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_268, 0), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %add_268), kwargs = {})
#   %where_33 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_33, %add_268, %mul_216), kwargs = {})
#   %add_269 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_33, %add_254), kwargs = {})
#   %add_270 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_269, %add_257), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
    tmp29 = tl.load(in_ptr9 + (x3), None)
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x3), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp20 + tmp27
    tmp30 = tmp29 > tmp16
    tmp32 = tmp31 * tmp29
    tmp33 = tl.where(tmp30, tmp29, tmp32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp28 + tmp35
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp36, None)
''', device_str='cuda')


# kernel path: inductor_cache/2t/c2tbfbikptbngqwhk2c4kv4hs7tqfht2e552yx63hz2xglsujtar.py
# Topologically Sorted Source Nodes: [output_79, long4_2, batch_norm_37, output_85, output4_4, output4_add_5], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_37 => add_280, mul_222, mul_223, sub_37
#   long4_2 => add_257
#   output4_4 => add_281
#   output4_add_5 => add_282
#   output_79 => gt_32, mul_208, where_32
#   output_85 => gt_34, mul_224, where_34
# Graph fragment:
#   %gt_32 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_256, 0), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_32, %add_256), kwargs = {})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %add_256, %mul_208), kwargs = {})
#   %add_257 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_32, %add_242), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_164, %unsqueeze_373), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_375), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_222, %unsqueeze_377), kwargs = {})
#   %add_280 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %unsqueeze_379), kwargs = {})
#   %gt_34 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_280, 0), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %add_280), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %add_280, %mul_224), kwargs = {})
#   %add_281 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_34, %add_269), kwargs = {})
#   %add_282 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_281, %add_257), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_out_ptr0 + (x3), None)
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x3), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp23 > tmp16
    tmp26 = tmp25 * tmp23
    tmp27 = tl.where(tmp24, tmp23, tmp26)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45g3dirtav6dlnlxgx34asetg6jxwiqx6cwnalfngo327ecpu2d.py
# Topologically Sorted Source Nodes: [output_85, output4_4, batch_norm_38, output_88, output4_5], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_38 => add_292, mul_230, mul_231, sub_38
#   output4_4 => add_281
#   output4_5 => add_293
#   output_85 => gt_34, mul_224, where_34
#   output_88 => gt_35, mul_232, where_35
# Graph fragment:
#   %gt_34 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_280, 0), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %add_280), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %add_280, %mul_224), kwargs = {})
#   %add_281 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_34, %add_269), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_171, %unsqueeze_385), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_387), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %unsqueeze_389), kwargs = {})
#   %add_292 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, %unsqueeze_391), kwargs = {})
#   %gt_35 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_292, 0), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, %add_292), kwargs = {})
#   %where_35 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %add_292, %mul_232), kwargs = {})
#   %add_293 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_35, %add_281), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), None)
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None)
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
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp22 = tmp21 > tmp16
    tmp24 = tmp23 * tmp21
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp20 + tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/ql/cqlhcne4sjxnifcrknyp2d4adqd2iho34hzwptna32ayn6c2ky65.py
# Topologically Sorted Source Nodes: [conv2d_172, up4_conv4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_172 => convolution_172
#   up4_conv4 => add_296, mul_234, mul_235, sub_39
# Graph fragment:
#   %convolution_172 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_293, %primals_366, %primals_367, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_172, %unsqueeze_393), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_395), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %unsqueeze_397), kwargs = {})
#   %add_296 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %unsqueeze_399), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/z4/cz4jclpu4bs2imgrmntwdhxi2j4u4oms5in6cwmch5lm7hbavfxl.py
# Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   up4_1 => convert_element_type_81
# Graph fragment:
#   %convert_element_type_81 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_37, torch.int64), kwargs = {})
triton_poi_fused__to_copy_46 = async_compile.triton('triton_poi_fused__to_copy_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/lm/clm46fiu7eviawum6hwsv5v4elugo3i6szuggdmgxeqgxvi6jbbs.py
# Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up4_1 => add_298, clamp_max
# Graph fragment:
#   %add_298 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_81, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_298, 3), kwargs = {})
triton_poi_fused_add_clamp_47 = async_compile.triton('triton_poi_fused_add_clamp_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/lk/clk6pglyqznp2avhye7hdfsbebf3quve3hhhpzheqeaanuekep2i.py
# Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up4_1 => add_297, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_80, iota, mul_237, sub_40, sub_42
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_80, 0.5), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_297, 0.5), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_237, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_40, 0.0), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_83), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_42, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wf/cwfzymplu3u4p2kzdybv6gjprturu7sr34iodf63jq5vcjeetuer.py
# Topologically Sorted Source Nodes: [up4, up4_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   up4 => gt_36, mul_236, where_36
#   up4_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_301, add_302, add_303, mul_239, mul_240, mul_241, sub_43, sub_44, sub_46
# Graph fragment:
#   %gt_36 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_296, 0), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_36, %add_296), kwargs = {})
#   %where_36 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_36, %add_296, %mul_236), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_36, [None, None, %convert_element_type_81, %convert_element_type_83]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_36, [None, None, %convert_element_type_81, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_36, [None, None, %clamp_max, %convert_element_type_83]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_36, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %clamp_max_2), kwargs = {})
#   %add_301 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_239), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %clamp_max_2), kwargs = {})
#   %add_302 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_240), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_302, %add_301), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %clamp_max_3), kwargs = {})
#   %add_303 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_301, %mul_241), kwargs = {})
triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_49 = async_compile.triton('triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x6 = xindex // 64
    x2 = ((xindex // 64) % 64)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x6), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp13 = tmp12 * tmp9
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr2 + (tmp18 + 4*tmp4 + 16*x6), None, eviction_policy='evict_last')
    tmp20 = tmp19 > tmp10
    tmp21 = tmp12 * tmp19
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp22 - tmp14
    tmp25 = tmp23 * tmp24
    tmp26 = tmp14 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (tmp8 + 4*tmp30 + 16*x6), None, eviction_policy='evict_last')
    tmp32 = tmp31 > tmp10
    tmp33 = tmp12 * tmp31
    tmp34 = tl.where(tmp32, tmp31, tmp33)
    tmp35 = tl.load(in_ptr2 + (tmp18 + 4*tmp30 + 16*x6), None, eviction_policy='evict_last')
    tmp36 = tmp35 > tmp10
    tmp37 = tmp12 * tmp35
    tmp38 = tl.where(tmp36, tmp35, tmp37)
    tmp39 = tmp38 - tmp34
    tmp40 = tmp39 * tmp24
    tmp41 = tmp34 + tmp40
    tmp42 = tmp41 - tmp26
    tmp44 = tmp42 * tmp43
    tmp45 = tmp26 + tmp44
    tl.store(in_out_ptr0 + (x4), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/ci/ccibasn2dcxtbvxhsq77npjd4jqogdn5fsgn34et6u5rnjrj7k5q.py
# Topologically Sorted Source Nodes: [output_90, output_91, conv2d_176, up3_conv3, add_218], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
# Source node to ATen node mapping:
#   add_218 => add_309
#   conv2d_176 => convolution_176
#   output_90 => add_304
#   output_91 => add_306, mul_243, mul_244, sub_47
#   up3_conv3 => add_308, mul_246, mul_247, sub_48
# Graph fragment:
#   %add_304 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_174, %convolution_175), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_304, %unsqueeze_401), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_403), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_243, %unsqueeze_405), kwargs = {})
#   %add_306 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_244, %unsqueeze_407), kwargs = {})
#   %convolution_176 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_195, %primals_380, %primals_381, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_176, %unsqueeze_409), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_411), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_246, %unsqueeze_413), kwargs = {})
#   %add_308 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %unsqueeze_415), kwargs = {})
#   %add_309 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_306, %add_308), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_50(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = ((xindex // 64) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr1 + (x0), None)
    tmp4 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(in_out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr0 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/nw/cnw2yqbdbp56qkxanlyzcgkzlfcnimxfl4wlnclyumzkvsevysso.py
# Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   up3_1 => convert_element_type_89
# Graph fragment:
#   %convert_element_type_89 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.int64), kwargs = {})
triton_poi_fused__to_copy_51 = async_compile.triton('triton_poi_fused__to_copy_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/x2/cx2dgellkfe64v7iuqrn7nulwjvmzydhpkb3albkvih5newmeh5y.py
# Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up3_1 => add_311, clamp_max_4
# Graph fragment:
#   %add_311 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_89, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_311, 7), kwargs = {})
triton_poi_fused_add_clamp_52 = async_compile.triton('triton_poi_fused_add_clamp_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_52(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/f2/cf2niixxtqznrdgexgrz4ynzl5wudiiwrwu5i4svkn3tzq5f72qx.py
# Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up3_1 => add_310, clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_88, iota_2, mul_249, sub_49, sub_51
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_88 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_310 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_88, 0.5), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_310, 0.5), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_249, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_49, 0.0), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_91), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_51, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ha/cha7uz57f2qhd3msrf6ibgwttcdol6g6exegnjnvbmvvofel3b6u.py
# Topologically Sorted Source Nodes: [up3, up3_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   up3 => gt_37, mul_248, where_37
#   up3_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_314, add_315, add_316, mul_251, mul_252, mul_253, sub_52, sub_53, sub_55
# Graph fragment:
#   %gt_37 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_309, 0), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_39, %add_309), kwargs = {})
#   %where_37 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %add_309, %mul_248), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_37, [None, None, %convert_element_type_89, %convert_element_type_91]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_37, [None, None, %convert_element_type_89, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_37, [None, None, %clamp_max_4, %convert_element_type_91]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_37, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %clamp_max_6), kwargs = {})
#   %add_314 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_251), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %clamp_max_6), kwargs = {})
#   %add_315 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_252), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_315, %add_314), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %clamp_max_7), kwargs = {})
#   %add_316 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_314, %mul_253), kwargs = {})
triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_54 = async_compile.triton('triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x6 = xindex // 256
    x2 = ((xindex // 256) % 32)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x6), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp13 = tmp12 * tmp9
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr2 + (tmp18 + 8*tmp4 + 64*x6), None, eviction_policy='evict_last')
    tmp20 = tmp19 > tmp10
    tmp21 = tmp12 * tmp19
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp22 - tmp14
    tmp25 = tmp23 * tmp24
    tmp26 = tmp14 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (tmp8 + 8*tmp30 + 64*x6), None, eviction_policy='evict_last')
    tmp32 = tmp31 > tmp10
    tmp33 = tmp12 * tmp31
    tmp34 = tl.where(tmp32, tmp31, tmp33)
    tmp35 = tl.load(in_ptr2 + (tmp18 + 8*tmp30 + 64*x6), None, eviction_policy='evict_last')
    tmp36 = tmp35 > tmp10
    tmp37 = tmp12 * tmp35
    tmp38 = tl.where(tmp36, tmp35, tmp37)
    tmp39 = tmp38 - tmp34
    tmp40 = tmp39 * tmp24
    tmp41 = tmp34 + tmp40
    tmp42 = tmp41 - tmp26
    tmp44 = tmp42 * tmp43
    tmp45 = tmp26 + tmp44
    tl.store(in_out_ptr0 + (x4), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/y5/cy5nxbv6ao26zcjv5fdwnfzu4x6bqe2y6criczdscze2aph2kxqf.py
# Topologically Sorted Source Nodes: [output_93, output_94, conv2d_180, up2_conv2, add_220], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
# Source node to ATen node mapping:
#   add_220 => add_322
#   conv2d_180 => convolution_180
#   output_93 => add_317
#   output_94 => add_319, mul_255, mul_256, sub_56
#   up2_conv2 => add_321, mul_258, mul_259, sub_57
# Graph fragment:
#   %add_317 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_178, %convolution_179), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_317, %unsqueeze_417), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_419), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_255, %unsqueeze_421), kwargs = {})
#   %add_319 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_256, %unsqueeze_423), kwargs = {})
#   %convolution_180 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_70, %primals_394, %primals_395, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_180, %unsqueeze_425), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_427), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %unsqueeze_429), kwargs = {})
#   %add_321 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %unsqueeze_431), kwargs = {})
#   %add_322 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_319, %add_321), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_55(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = ((xindex // 256) % 24)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr1 + (x0), None)
    tmp4 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(in_out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr0 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/sx/csxmoptarxiv2msy57iviq4j4g2dvmre6tdodii73qbbhlgyjrhf.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   up2_1 => convert_element_type_97
# Graph fragment:
#   %convert_element_type_97 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_43, torch.int64), kwargs = {})
triton_poi_fused__to_copy_56 = async_compile.triton('triton_poi_fused__to_copy_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_56(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4j/c4jzr3h66m2ybpu2rt3conifkn5q3ibr5jpuhucekv4s5hd2o4pu.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => add_324, clamp_max_8
# Graph fragment:
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_97, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_324, 15), kwargs = {})
triton_poi_fused_add_clamp_57 = async_compile.triton('triton_poi_fused_add_clamp_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_57(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5h/c5hizeyj6m2wdbvqc2e5fiofdntwaynq7pzuvhxruwkh5qfitxfu.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => add_323, clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_96, iota_4, mul_261, sub_58, sub_60
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_96 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_96, 0.5), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_323, 0.5), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_261, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_58, 0.0), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_99), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_60, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/37/c37qrzgooofvyabvzyfidna2y3i6zeo325uhl7x6q46grg5locak.py
# Topologically Sorted Source Nodes: [up2, up2_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   up2 => gt_38, mul_260, where_38
#   up2_1 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_327, add_328, add_329, mul_263, mul_264, mul_265, sub_61, sub_62, sub_64
# Graph fragment:
#   %gt_38 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_322, 0), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_42, %add_322), kwargs = {})
#   %where_38 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %add_322, %mul_260), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_38, [None, None, %convert_element_type_97, %convert_element_type_99]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_38, [None, None, %convert_element_type_97, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_38, [None, None, %clamp_max_8, %convert_element_type_99]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_38, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %clamp_max_10), kwargs = {})
#   %add_327 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_263), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %clamp_max_10), kwargs = {})
#   %add_328 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_264), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_328, %add_327), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %clamp_max_11), kwargs = {})
#   %add_329 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_327, %mul_265), kwargs = {})
triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_59 = async_compile.triton('triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_59', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x6 = xindex // 1024
    x2 = ((xindex // 1024) % 24)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x6), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp13 = tmp12 * tmp9
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr2 + (tmp18 + 16*tmp4 + 256*x6), None, eviction_policy='evict_last')
    tmp20 = tmp19 > tmp10
    tmp21 = tmp12 * tmp19
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp22 - tmp14
    tmp25 = tmp23 * tmp24
    tmp26 = tmp14 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (tmp8 + 16*tmp30 + 256*x6), None, eviction_policy='evict_last')
    tmp32 = tmp31 > tmp10
    tmp33 = tmp12 * tmp31
    tmp34 = tl.where(tmp32, tmp31, tmp33)
    tmp35 = tl.load(in_ptr2 + (tmp18 + 16*tmp30 + 256*x6), None, eviction_policy='evict_last')
    tmp36 = tmp35 > tmp10
    tmp37 = tmp12 * tmp35
    tmp38 = tl.where(tmp36, tmp35, tmp37)
    tmp39 = tmp38 - tmp34
    tmp40 = tmp39 * tmp24
    tmp41 = tmp34 + tmp40
    tmp42 = tmp41 - tmp26
    tmp44 = tmp42 * tmp43
    tmp45 = tmp26 + tmp44
    tl.store(in_out_ptr0 + (x4), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/rn/crnd6r6hwhhtdelu5dqbpvxlguyuwv4w7og3hql4xjpboccw46bp.py
# Topologically Sorted Source Nodes: [output_96, output_97, conv2d_184, up1_conv1, add_222, up1], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution, aten._prelu_kernel]
# Source node to ATen node mapping:
#   add_222 => add_335
#   conv2d_184 => convolution_184
#   output_96 => add_330
#   output_97 => add_332, mul_267, mul_268, sub_65
#   up1 => gt_39, mul_272, where_39
#   up1_conv1 => add_334, mul_270, mul_271, sub_66
# Graph fragment:
#   %add_330 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_182, %convolution_183), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_330, %unsqueeze_433), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_435), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %unsqueeze_437), kwargs = {})
#   %add_332 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_268, %unsqueeze_439), kwargs = {})
#   %convolution_184 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_14, %primals_408, %primals_409, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_184, %unsqueeze_441), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_443), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %unsqueeze_445), kwargs = {})
#   %add_334 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %unsqueeze_447), kwargs = {})
#   %add_335 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_332, %add_334), kwargs = {})
#   %gt_39 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_335, 0), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %add_335), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_39, %add_335, %mul_272), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_60', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_60(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = ((xindex // 1024) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr1 + (x0), None)
    tmp4 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
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
    tmp34 = 0.0
    tmp35 = tmp33 > tmp34
    tmp37 = tmp36 * tmp33
    tmp38 = tl.where(tmp35, tmp33, tmp37)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(in_out_ptr1 + (x0), tmp5, None)
    tl.store(in_out_ptr2 + (x0), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpfeeeeqpngc3w36vbddl2ltm45zqk46ei5lrwlw5nn7k7g66d7q.py
# Topologically Sorted Source Nodes: [pred1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   pred1 => convert_element_type_105
# Graph fragment:
#   %convert_element_type_105 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_46, torch.int64), kwargs = {})
triton_poi_fused__to_copy_61 = async_compile.triton('triton_poi_fused__to_copy_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_61(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ra/craadugxkaanf2py4cvedg3m67hqfohmy3gesgljvvou3x4muo27.py
# Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   pred1 => add_337, clamp_max_12
# Graph fragment:
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_105, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_337, 31), kwargs = {})
triton_poi_fused_add_clamp_62 = async_compile.triton('triton_poi_fused_add_clamp_62', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_62(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/c4/cc45pdmrxtuz52hmiggwyd54xrhfrxrp75nfsd6f6gdpbpwiyfbm.py
# Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   pred1 => add_336, clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_104, iota_6, mul_273, sub_67, sub_69
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_104 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_104, 0.5), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_336, 0.5), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_273, 0.5), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_67, 0.0), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_107), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_69, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5t/c5tqqzbkdx3i2hx75qtnbc2r7nkpi2larti4zlcwdqdp4rprio6g.py
# Topologically Sorted Source Nodes: [input_8, pred1], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_8 => convolution_185
#   pred1 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_340, add_341, add_342, mul_275, mul_276, mul_277, sub_70, sub_71, sub_73
# Graph fragment:
#   %convolution_185 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%where_39, %primals_415, %primals_416, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_185, [None, None, %convert_element_type_105, %convert_element_type_107]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_185, [None, None, %convert_element_type_105, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_185, [None, None, %clamp_max_12, %convert_element_type_107]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_185, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %clamp_max_14), kwargs = {})
#   %add_340 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_275), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %clamp_max_14), kwargs = {})
#   %add_341 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_276), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_341, %add_340), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %clamp_max_15), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_340, %mul_277), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sub_64 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sub_64', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sub_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sub_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x6 = xindex // 4096
    x2 = ((xindex // 4096) % 2)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x6), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tl.load(in_ptr2 + (tmp15 + 32*tmp4 + 1024*x6), None, eviction_policy='evict_last')
    tmp17 = tmp16 + tmp10
    tmp18 = tmp17 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 + tmp1
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tmp26 = tl.load(in_ptr2 + (tmp8 + 32*tmp25 + 1024*x6), None, eviction_policy='evict_last')
    tmp27 = tmp26 + tmp10
    tmp28 = tl.load(in_ptr2 + (tmp15 + 32*tmp25 + 1024*x6), None, eviction_policy='evict_last')
    tmp29 = tmp28 + tmp10
    tmp30 = tmp29 - tmp27
    tmp31 = tmp30 * tmp19
    tmp32 = tmp27 + tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp33 * tmp34
    tmp36 = tmp21 + tmp35
    tl.store(in_out_ptr0 + (x4), tmp36, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416 = args
    args.clear()
    assert_size_stride(primals_1, (8, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (8, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_4, (8, ), (1, ))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (8, ), (1, ))
    assert_size_stride(primals_8, (8, ), (1, ))
    assert_size_stride(primals_9, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, ), (1, ))
    assert_size_stride(primals_15, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (8, ), (1, ))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (8, ), (1, ))
    assert_size_stride(primals_21, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_22, (8, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_23, (8, ), (1, ))
    assert_size_stride(primals_24, (8, ), (1, ))
    assert_size_stride(primals_25, (8, ), (1, ))
    assert_size_stride(primals_26, (8, ), (1, ))
    assert_size_stride(primals_27, (8, ), (1, ))
    assert_size_stride(primals_28, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_29, (8, ), (1, ))
    assert_size_stride(primals_30, (8, ), (1, ))
    assert_size_stride(primals_31, (8, ), (1, ))
    assert_size_stride(primals_32, (8, ), (1, ))
    assert_size_stride(primals_33, (8, ), (1, ))
    assert_size_stride(primals_34, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (16, ), (1, ))
    assert_size_stride(primals_39, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_40, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_41, (24, ), (1, ))
    assert_size_stride(primals_42, (24, ), (1, ))
    assert_size_stride(primals_43, (24, ), (1, ))
    assert_size_stride(primals_44, (24, ), (1, ))
    assert_size_stride(primals_45, (24, ), (1, ))
    assert_size_stride(primals_46, (6, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_47, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_48, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (4, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_52, (24, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_53, (24, ), (1, ))
    assert_size_stride(primals_54, (24, ), (1, ))
    assert_size_stride(primals_55, (24, ), (1, ))
    assert_size_stride(primals_56, (24, ), (1, ))
    assert_size_stride(primals_57, (24, ), (1, ))
    assert_size_stride(primals_58, (6, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_59, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_60, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_61, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_62, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_63, (4, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_64, (24, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_65, (24, ), (1, ))
    assert_size_stride(primals_66, (24, ), (1, ))
    assert_size_stride(primals_67, (24, ), (1, ))
    assert_size_stride(primals_68, (24, ), (1, ))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_71, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_72, (24, ), (1, ))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_75, (24, ), (1, ))
    assert_size_stride(primals_76, (24, ), (1, ))
    assert_size_stride(primals_77, (6, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_78, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_79, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_81, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_82, (4, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_83, (24, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_84, (24, ), (1, ))
    assert_size_stride(primals_85, (24, ), (1, ))
    assert_size_stride(primals_86, (24, ), (1, ))
    assert_size_stride(primals_87, (24, ), (1, ))
    assert_size_stride(primals_88, (24, ), (1, ))
    assert_size_stride(primals_89, (6, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_90, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_91, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_93, (6, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_94, (4, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_95, (24, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_97, (24, ), (1, ))
    assert_size_stride(primals_98, (24, ), (1, ))
    assert_size_stride(primals_99, (24, ), (1, ))
    assert_size_stride(primals_100, (24, ), (1, ))
    assert_size_stride(primals_101, (48, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_102, (48, ), (1, ))
    assert_size_stride(primals_103, (48, ), (1, ))
    assert_size_stride(primals_104, (48, ), (1, ))
    assert_size_stride(primals_105, (48, ), (1, ))
    assert_size_stride(primals_106, (32, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_107, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, ), (1, ))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_114, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_115, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_117, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_119, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (32, ), (1, ))
    assert_size_stride(primals_125, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_126, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_129, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_131, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_132, (32, ), (1, ))
    assert_size_stride(primals_133, (32, ), (1, ))
    assert_size_stride(primals_134, (32, ), (1, ))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_138, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (32, ), (1, ))
    assert_size_stride(primals_141, (32, ), (1, ))
    assert_size_stride(primals_142, (32, ), (1, ))
    assert_size_stride(primals_143, (32, ), (1, ))
    assert_size_stride(primals_144, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_145, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_147, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_148, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_150, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_151, (32, ), (1, ))
    assert_size_stride(primals_152, (32, ), (1, ))
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (32, ), (1, ))
    assert_size_stride(primals_155, (32, ), (1, ))
    assert_size_stride(primals_156, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_157, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_158, (32, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_164, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_166, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_169, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_170, (32, ), (1, ))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (32, ), (1, ))
    assert_size_stride(primals_173, (32, ), (1, ))
    assert_size_stride(primals_174, (32, ), (1, ))
    assert_size_stride(primals_175, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_176, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_177, (32, ), (1, ))
    assert_size_stride(primals_178, (32, ), (1, ))
    assert_size_stride(primals_179, (32, ), (1, ))
    assert_size_stride(primals_180, (32, ), (1, ))
    assert_size_stride(primals_181, (32, ), (1, ))
    assert_size_stride(primals_182, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_183, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_184, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_185, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_186, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_187, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_188, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_195, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_202, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_203, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_204, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_205, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_207, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, ), (1, ))
    assert_size_stride(primals_213, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_214, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_215, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_216, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_217, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_219, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_220, (32, ), (1, ))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (32, ), (1, ))
    assert_size_stride(primals_223, (32, ), (1, ))
    assert_size_stride(primals_224, (32, ), (1, ))
    assert_size_stride(primals_225, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_226, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_228, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_229, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_230, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_231, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (32, ), (1, ))
    assert_size_stride(primals_234, (32, ), (1, ))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, ), (1, ))
    assert_size_stride(primals_237, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_238, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_239, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_241, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_242, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_243, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_244, (32, ), (1, ))
    assert_size_stride(primals_245, (32, ), (1, ))
    assert_size_stride(primals_246, (32, ), (1, ))
    assert_size_stride(primals_247, (32, ), (1, ))
    assert_size_stride(primals_248, (32, ), (1, ))
    assert_size_stride(primals_249, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_255, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, ), (1, ))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, ), (1, ))
    assert_size_stride(primals_261, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_262, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_263, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_264, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_265, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_266, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_267, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_268, (64, ), (1, ))
    assert_size_stride(primals_269, (64, ), (1, ))
    assert_size_stride(primals_270, (64, ), (1, ))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_274, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_275, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_276, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_277, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_278, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_279, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, ), (1, ))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (64, ), (1, ))
    assert_size_stride(primals_285, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_286, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_287, (64, ), (1, ))
    assert_size_stride(primals_288, (64, ), (1, ))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, ), (1, ))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_293, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_295, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_296, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_297, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_298, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, ), (1, ))
    assert_size_stride(primals_303, (64, ), (1, ))
    assert_size_stride(primals_304, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_305, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_306, (64, ), (1, ))
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_312, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_313, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_314, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_315, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_316, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_317, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, ), (1, ))
    assert_size_stride(primals_321, (64, ), (1, ))
    assert_size_stride(primals_322, (64, ), (1, ))
    assert_size_stride(primals_323, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_324, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_325, (64, ), (1, ))
    assert_size_stride(primals_326, (64, ), (1, ))
    assert_size_stride(primals_327, (64, ), (1, ))
    assert_size_stride(primals_328, (64, ), (1, ))
    assert_size_stride(primals_329, (64, ), (1, ))
    assert_size_stride(primals_330, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_331, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_332, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_333, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_334, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_335, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_336, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (64, ), (1, ))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_343, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_344, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_345, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_347, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_348, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (64, ), (1, ))
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (64, ), (1, ))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_355, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_356, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_357, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_358, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_359, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_360, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (64, ), (1, ))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, ), (1, ))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (64, ), (1, ))
    assert_size_stride(primals_371, (64, ), (1, ))
    assert_size_stride(primals_372, (64, ), (1, ))
    assert_size_stride(primals_373, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_374, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_375, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_376, (32, ), (1, ))
    assert_size_stride(primals_377, (32, ), (1, ))
    assert_size_stride(primals_378, (32, ), (1, ))
    assert_size_stride(primals_379, (32, ), (1, ))
    assert_size_stride(primals_380, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_381, (32, ), (1, ))
    assert_size_stride(primals_382, (32, ), (1, ))
    assert_size_stride(primals_383, (32, ), (1, ))
    assert_size_stride(primals_384, (32, ), (1, ))
    assert_size_stride(primals_385, (32, ), (1, ))
    assert_size_stride(primals_386, (32, ), (1, ))
    assert_size_stride(primals_387, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_388, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_389, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_390, (24, ), (1, ))
    assert_size_stride(primals_391, (24, ), (1, ))
    assert_size_stride(primals_392, (24, ), (1, ))
    assert_size_stride(primals_393, (24, ), (1, ))
    assert_size_stride(primals_394, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_395, (24, ), (1, ))
    assert_size_stride(primals_396, (24, ), (1, ))
    assert_size_stride(primals_397, (24, ), (1, ))
    assert_size_stride(primals_398, (24, ), (1, ))
    assert_size_stride(primals_399, (24, ), (1, ))
    assert_size_stride(primals_400, (24, ), (1, ))
    assert_size_stride(primals_401, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_402, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_403, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_404, (8, ), (1, ))
    assert_size_stride(primals_405, (8, ), (1, ))
    assert_size_stride(primals_406, (8, ), (1, ))
    assert_size_stride(primals_407, (8, ), (1, ))
    assert_size_stride(primals_408, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_409, (8, ), (1, ))
    assert_size_stride(primals_410, (8, ), (1, ))
    assert_size_stride(primals_411, (8, ), (1, ))
    assert_size_stride(primals_412, (8, ), (1, ))
    assert_size_stride(primals_413, (8, ), (1, ))
    assert_size_stride(primals_414, (8, ), (1, ))
    assert_size_stride(primals_415, (2, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_416, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 8, 64, 64), (32768, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf1, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_2, primals_9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf2 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm, output_1, batch_norm_1, output_2, output1_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_0.run(buf1, primals_4, primals_5, primals_6, primals_7, buf3, primals_10, primals_11, primals_12, primals_13, primals_14, primals_8, buf2, buf4, buf5, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf5, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_22, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf9, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf7 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_1, output_2, batch_norm_2, output_3, output1, batch_norm_3, output_5, long1, output1_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_1.run(buf6, primals_16, primals_17, primals_18, primals_19, buf9, primals_23, primals_24, primals_25, primals_26, primals_20, buf4, primals_14, primals_27, buf2, primals_8, buf7, buf10, buf11, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf13 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [output_2, output_3, output1, batch_norm_4, output_6, output1_1], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_2.run(buf14, buf12, primals_29, primals_30, primals_31, primals_32, primals_33, buf7, primals_20, buf4, primals_14, 32768, grid=grid(32768), stream=stream0)
        buf15 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf10, primals_27, buf2, primals_8, buf14, buf15, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf17 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf16, primals_35, primals_36, primals_37, primals_38, buf17, 65536, grid=grid(65536), stream=stream0)
        del primals_38
        buf18 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [output_1, output_5, long1, x1, add_6], Original ATen: [aten._prelu_kernel, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_clone_5.run(buf18, buf17, primals_27, buf2, primals_8, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 24, 32, 32), (24576, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [output_7], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_40, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf20, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf22 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x2, add_7], Original ATen: [aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_6.run(buf17, buf14, buf22, 32768, grid=grid(32768), stream=stream0)
        del buf17
        # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 6, 32, 32), (6144, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [d2], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf23, primals_48, stride=(2, 2), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf25, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d3], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf23, primals_49, stride=(2, 2), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf26, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d4], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf23, primals_50, stride=(2, 2), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf27, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d1], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_47, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf24, (4, 6, 16, 16), (1536, 256, 16, 1))
        buf29 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [p, d1_1], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_7.run(buf29, buf23, 6144, grid=grid(6144), stream=stream0)
        buf30 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf29, buf25, buf26, buf27, buf30, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf31, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf32 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf29, buf31, buf25, buf26, buf27, buf32, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_10], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf33, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf21 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf34 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf35 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_6, output_8, batch_norm_7, output_11, output2_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_10.run(buf20, primals_41, primals_42, primals_43, primals_44, buf33, primals_53, primals_54, primals_55, primals_56, primals_57, primals_45, buf21, buf34, buf35, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_12], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d1_3], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf37, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d2_3], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf36, primals_60, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf38, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d3_3], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf36, primals_61, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf39, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d4_3], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf36, primals_62, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf40, (4, 6, 16, 16), (1536, 256, 16, 1))
        buf42 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [p_1, d1_4], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_11.run(buf42, buf36, 6144, grid=grid(6144), stream=stream0)
        buf43 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf42, buf38, buf39, buf40, buf43, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_63, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf44, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf45 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf42, buf44, buf38, buf39, buf40, buf45, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_13], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf46, (4, 24, 16, 16), (6144, 256, 16, 1))
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf35, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 24, 16, 16), (6144, 256, 16, 1))
        # Topologically Sorted Source Nodes: [output_15], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_71, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf49, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf47 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf50 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf51 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_8, output_11, batch_norm_8, output_14, output2, batch_norm_9, output_16, long2, output2_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_12.run(buf46, primals_65, primals_66, primals_67, primals_68, buf49, primals_72, primals_73, primals_74, primals_75, primals_69, buf34, primals_57, primals_76, buf21, primals_45, buf47, buf50, buf51, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_17], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d1_6], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf53, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d2_6], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf52, primals_79, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf54, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d3_6], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf52, primals_80, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf55, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d4_6], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf52, primals_81, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf56, (4, 6, 16, 16), (1536, 256, 16, 1))
        buf58 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [p_2, d1_7], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_11.run(buf58, buf52, 6144, grid=grid(6144), stream=stream0)
        buf59 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf58, buf54, buf55, buf56, buf59, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf60, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf61 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf58, buf60, buf54, buf55, buf56, buf61, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_18], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_83, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf62, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf63 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_8, output_11, output_14, output2, output_16, long2, batch_norm_10, output_19, output2_1, output2_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_13.run(buf64, buf62, primals_84, primals_85, primals_86, primals_87, primals_88, buf47, primals_69, buf34, primals_57, buf50, primals_76, buf21, primals_45, buf65, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_20], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d1_9], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf67, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d2_9], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf66, primals_91, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf68, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d3_9], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf66, primals_92, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf69, (4, 6, 16, 16), (1536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d4_9], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf66, primals_93, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf70, (4, 6, 16, 16), (1536, 256, 16, 1))
        buf72 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [p_3, d1_10], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_11.run(buf72, buf66, 6144, grid=grid(6144), stream=stream0)
        buf73 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf72, buf68, buf69, buf70, buf73, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf74, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf75 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf72, buf74, buf68, buf69, buf70, buf75, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [output_21], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf76, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf77 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_11, output_22, output2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_14.run(buf78, buf76, primals_96, primals_97, primals_98, primals_99, primals_100, buf64, 24576, grid=grid(24576), stream=stream0)
        buf79 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf50, primals_76, buf21, primals_45, buf78, buf79, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf81 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf80, primals_102, primals_103, primals_104, primals_105, buf81, 49152, grid=grid(49152), stream=stream0)
        del primals_105
        buf82 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [output_8, output_16, long2, x1_1, add_48], Original ATen: [aten._prelu_kernel, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_clone_17.run(buf82, buf81, primals_76, buf21, primals_45, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 32, 16, 16), (8192, 256, 16, 1))
        # Topologically Sorted Source Nodes: [output_23], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_107, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf84, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf86 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x2_1, add_49], Original ATen: [aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_18.run(buf81, buf78, buf86, 24576, grid=grid(24576), stream=stream0)
        del buf81
        # Topologically Sorted Source Nodes: [output_25], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 8, 16, 16), (2048, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d2_12], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf87, primals_115, stride=(2, 2), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf89, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_12], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf87, primals_116, stride=(2, 2), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf90, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_12], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf87, primals_117, stride=(2, 2), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf91, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_12], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_114, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf88, (4, 8, 8, 8), (512, 64, 8, 1))
        buf93 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [p_4, d1_13], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_19.run(buf93, buf87, 2048, grid=grid(2048), stream=stream0)
        buf94 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf93, buf89, buf90, buf91, buf94, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf95, (4, 4, 8, 8), (256, 64, 8, 1))
        buf96 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf93, buf95, buf89, buf90, buf91, buf96, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_26], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf97, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf85 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf98 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf99 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_13, output_24, batch_norm_14, output_27, output3_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_22.run(buf84, primals_108, primals_109, primals_110, primals_111, buf97, primals_120, primals_121, primals_122, primals_123, primals_124, primals_112, buf85, buf98, buf99, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_28], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_15], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf101, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_15], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf100, primals_127, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf102, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_15], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf100, primals_128, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf103, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_15], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf100, primals_129, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf104, (4, 8, 8, 8), (512, 64, 8, 1))
        buf106 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [p_5, d1_16], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf106, buf100, 2048, grid=grid(2048), stream=stream0)
        buf107 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf106, buf102, buf103, buf104, buf107, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf108, (4, 4, 8, 8), (256, 64, 8, 1))
        buf109 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf106, buf108, buf102, buf103, buf104, buf109, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_29], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf110, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf99, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [output_31], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_138, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf113, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf111 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf114 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf115 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_24, output_27, batch_norm_15, output_30, output3, batch_norm_16, output_32, long3, output3_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_24.run(buf110, primals_132, primals_133, primals_134, primals_135, buf113, primals_139, primals_140, primals_141, primals_142, primals_136, buf98, primals_124, primals_143, buf85, primals_112, buf111, buf114, buf115, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_33], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_18], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf117, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_18], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf116, primals_146, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf118, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_18], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf116, primals_147, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf119, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_18], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf116, primals_148, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf120, (4, 8, 8, 8), (512, 64, 8, 1))
        buf122 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [p_6, d1_19], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf122, buf116, 2048, grid=grid(2048), stream=stream0)
        buf123 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf122, buf118, buf119, buf120, buf123, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf124, (4, 4, 8, 8), (256, 64, 8, 1))
        buf125 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf122, buf124, buf118, buf119, buf120, buf125, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_34], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf126, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf115, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [output_36], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_157, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf130, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf127 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf128 = buf127; del buf127  # reuse
        buf131 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_24, output_27, output_30, output3, output_32, long3, batch_norm_17, output_35, output3_1, batch_norm_18, output_37, long3_1, output3_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_25.run(buf128, buf132, buf126, primals_151, primals_152, primals_153, primals_154, primals_155, buf111, primals_136, buf98, primals_124, buf130, primals_158, primals_159, primals_160, primals_161, primals_162, buf114, primals_143, buf85, primals_112, buf133, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_38], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_21], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf135, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_21], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf134, primals_165, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf136, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_21], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf134, primals_166, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf137, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_21], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf134, primals_167, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf138, (4, 8, 8, 8), (512, 64, 8, 1))
        buf140 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [p_7, d1_22], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf140, buf134, 2048, grid=grid(2048), stream=stream0)
        buf141 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf140, buf136, buf137, buf138, buf141, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf142, (4, 4, 8, 8), (256, 64, 8, 1))
        buf143 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [cat_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf140, buf142, buf136, buf137, buf138, buf143, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_39], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf144, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf133, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [output_41], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_176, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf147, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf145 = buf114; del buf114  # reuse
        buf148 = buf111; del buf111  # reuse
        buf149 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_19, output_40, output3_2, batch_norm_20, output_42, long3_2, output3_add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_26.run(buf144, primals_170, primals_171, primals_172, primals_173, buf147, primals_177, primals_178, primals_179, primals_180, primals_174, buf128, primals_181, buf132, buf145, buf148, buf149, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_43], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_24], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf151, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_24], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf150, primals_184, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf152, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_24], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf150, primals_185, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf153, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_24], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf150, primals_186, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf154, (4, 8, 8, 8), (512, 64, 8, 1))
        buf156 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [p_8, d1_25], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf156, buf150, 2048, grid=grid(2048), stream=stream0)
        buf157 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf156, buf152, buf153, buf154, buf157, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf158, (4, 4, 8, 8), (256, 64, 8, 1))
        buf159 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf156, buf158, buf152, buf153, buf154, buf159, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf160, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf149, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [output_46], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_195, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf164, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf161 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf162 = buf161; del buf161  # reuse
        buf222 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        buf166 = reinterpret_tensor(buf222, (4, 32, 8, 8), (4096, 64, 8, 1), 0)  # alias
        buf167 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_40, output3_2, output_42, long3_2, batch_norm_21, output_45, output3_3, batch_norm_22, output_47, long3_3, output3_add_4], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_27.run(buf162, buf160, primals_189, primals_190, primals_191, primals_192, primals_193, buf145, primals_174, buf128, buf164, primals_196, primals_197, primals_198, primals_199, primals_200, buf148, primals_181, buf132, buf166, buf167, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_48], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_27], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf169, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_27], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf168, primals_203, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf170, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_27], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf168, primals_204, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf171, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_27], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf168, primals_205, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf172, (4, 8, 8, 8), (512, 64, 8, 1))
        buf174 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [p_9, d1_28], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf174, buf168, 2048, grid=grid(2048), stream=stream0)
        buf175 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [cat_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf174, buf170, buf171, buf172, buf175, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_91], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf176, (4, 4, 8, 8), (256, 64, 8, 1))
        buf177 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [cat_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf174, buf176, buf170, buf171, buf172, buf177, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_49], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf178, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf179 = buf132; del buf132  # reuse
        buf180 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_23, output_50, output3_4, output3_add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28.run(buf178, primals_208, primals_209, primals_210, primals_211, primals_212, buf162, buf166, buf179, buf180, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_51], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_30], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf182, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_30], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf181, primals_215, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf183, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_30], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf181, primals_216, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf184, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_30], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf181, primals_217, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf185, (4, 8, 8, 8), (512, 64, 8, 1))
        buf187 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [p_10, d1_31], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf187, buf181, 2048, grid=grid(2048), stream=stream0)
        buf188 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf187, buf183, buf184, buf185, buf188, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_98], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf189, (4, 4, 8, 8), (256, 64, 8, 1))
        buf190 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf187, buf189, buf183, buf184, buf185, buf190, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_52], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf191, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf192 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_50, output3_4, batch_norm_24, output_53, output3_5, output3_add_6], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_29.run(buf193, buf191, primals_220, primals_221, primals_222, primals_223, primals_224, buf179, primals_212, buf162, buf166, buf194, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_54], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_33], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf196, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_33], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf195, primals_227, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf197, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_33], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf195, primals_228, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf198, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_33], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf195, primals_229, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf199, (4, 8, 8, 8), (512, 64, 8, 1))
        buf201 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [p_11, d1_34], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf201, buf195, 2048, grid=grid(2048), stream=stream0)
        buf202 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf201, buf197, buf198, buf199, buf202, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf203, (4, 4, 8, 8), (256, 64, 8, 1))
        buf204 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [cat_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf201, buf203, buf197, buf198, buf199, buf204, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_55], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf205, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf206 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf207 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_25, output_56, output3_6, output3_add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_28.run(buf205, primals_232, primals_233, primals_234, primals_235, primals_236, buf193, buf166, buf206, buf207, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_57], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_36], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf209, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_36], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf208, primals_239, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf210, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d3_36], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf208, primals_240, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf211, (4, 8, 8, 8), (512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d4_36], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf208, primals_241, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf212, (4, 8, 8, 8), (512, 64, 8, 1))
        buf214 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [p_12, d1_37], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_23.run(buf214, buf208, 2048, grid=grid(2048), stream=stream0)
        buf215 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf214, buf210, buf211, buf212, buf215, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_112], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf216, (4, 4, 8, 8), (256, 64, 8, 1))
        buf217 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf214, buf216, buf210, buf211, buf212, buf217, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_58], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf218, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf219 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf220 = buf219; del buf219  # reuse
        buf221 = reinterpret_tensor(buf222, (4, 32, 8, 8), (4096, 64, 8, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [output_56, output3_6, batch_norm_26, output_59, output3_7, cat_28], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_cat_30.run(buf220, buf218, primals_244, primals_245, primals_246, primals_247, primals_248, buf206, primals_236, buf193, buf221, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf224 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf223, primals_250, primals_251, primals_252, primals_253, buf224, 16384, grid=grid(16384), stream=stream0)
        del primals_253
        buf225 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x1_2, add_143], Original ATen: [aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_32.run(buf224, buf166, buf225, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_115], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 64, 8, 8), (4096, 64, 8, 1))
        # Topologically Sorted Source Nodes: [output_60], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_255, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf227, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf229 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x2_2, add_144], Original ATen: [aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_33.run(buf224, buf220, buf229, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [output_62], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_39], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf230, primals_263, stride=(2, 2), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf232, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_39], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf230, primals_264, stride=(2, 2), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf233, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_39], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf230, primals_265, stride=(2, 2), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf234, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_39], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_262, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf231, (4, 16, 4, 4), (256, 16, 4, 1))
        buf236 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [p_13, d1_40], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_34.run(buf236, buf230, 1024, grid=grid(1024), stream=stream0)
        buf237 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf236, buf232, buf233, buf234, buf237, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_122], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf238, (4, 4, 4, 4), (64, 16, 4, 1))
        buf239 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf236, buf238, buf232, buf233, buf234, buf239, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_63], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf240, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf228 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf241 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf242 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_28, output_61, batch_norm_29, output_64, output4_add], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_37.run(buf227, primals_256, primals_257, primals_258, primals_259, buf240, primals_268, primals_269, primals_270, primals_271, primals_272, primals_260, buf228, buf241, buf242, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_65], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_42], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf244, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_42], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf243, primals_275, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf245, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_42], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf243, primals_276, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf246, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_42], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf243, primals_277, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf247, (4, 16, 4, 4), (256, 16, 4, 1))
        buf249 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [p_14, d1_43], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf249, buf243, 1024, grid=grid(1024), stream=stream0)
        buf250 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf249, buf245, buf246, buf247, buf250, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_129], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf251, (4, 4, 4, 4), (64, 16, 4, 1))
        buf252 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf249, buf251, buf245, buf246, buf247, buf252, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_66], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf253, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_131], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf242, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [output_68], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_286, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf256, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf254 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf257 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf258 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_61, output_64, batch_norm_30, output_67, output4, batch_norm_31, output_69, long4, output4_add_1], Original ATen: [aten._prelu_kernel, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_39.run(buf253, primals_280, primals_281, primals_282, primals_283, buf256, primals_287, primals_288, primals_289, primals_290, primals_284, buf241, primals_272, primals_291, buf228, primals_260, buf254, buf257, buf258, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_70], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_45], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf260, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_45], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf259, primals_294, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf261, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_45], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf259, primals_295, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf262, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_45], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf259, primals_296, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf263, (4, 16, 4, 4), (256, 16, 4, 1))
        buf265 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [p_15, d1_46], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf265, buf259, 1024, grid=grid(1024), stream=stream0)
        buf266 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf265, buf261, buf262, buf263, buf266, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_138], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf267, (4, 4, 4, 4), (64, 16, 4, 1))
        buf268 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf265, buf267, buf261, buf262, buf263, buf268, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_71], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf269, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_140], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf258, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [output_73], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_305, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf273, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf270 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf271 = buf270; del buf270  # reuse
        buf274 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf275 = buf274; del buf274  # reuse
        buf276 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_61, output_64, output_67, output4, output_69, long4, batch_norm_32, output_72, output4_1, batch_norm_33, output_74, long4_1, output4_add_2], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_40.run(buf271, buf275, buf269, primals_299, primals_300, primals_301, primals_302, primals_303, buf254, primals_284, buf241, primals_272, buf273, primals_306, primals_307, primals_308, primals_309, primals_310, buf257, primals_291, buf228, primals_260, buf276, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_75], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_48], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf278, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_48], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf277, primals_313, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf279, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_48], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf277, primals_314, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf280, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_48], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf277, primals_315, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf281, (4, 16, 4, 4), (256, 16, 4, 1))
        buf283 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [p_16, d1_49], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf283, buf277, 1024, grid=grid(1024), stream=stream0)
        buf284 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [cat_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf283, buf279, buf280, buf281, buf284, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_147], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf285, (4, 4, 4, 4), (64, 16, 4, 1))
        buf286 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [cat_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf283, buf285, buf279, buf280, buf281, buf286, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_76], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf287, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_149], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf276, primals_323, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [output_78], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_324, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf290, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf288 = buf241; del buf241  # reuse
        buf291 = buf228; del buf228  # reuse
        buf292 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_34, output_77, output4_2, batch_norm_35, output_79, long4_2, output4_add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_41.run(buf287, primals_318, primals_319, primals_320, primals_321, buf290, primals_325, primals_326, primals_327, primals_328, primals_322, buf271, primals_329, buf275, buf288, buf291, buf292, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_80], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_51], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf294, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_51], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf293, primals_332, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf295, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_51], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf293, primals_333, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf296, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_51], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf293, primals_334, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf297, (4, 16, 4, 4), (256, 16, 4, 1))
        buf299 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [p_17, d1_52], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf299, buf293, 1024, grid=grid(1024), stream=stream0)
        buf300 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf299, buf295, buf296, buf297, buf300, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_156], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf301, (4, 4, 4, 4), (64, 16, 4, 1))
        buf302 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf299, buf301, buf295, buf296, buf297, buf302, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_81], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf303, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf304 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf305 = buf304; del buf304  # reuse
        buf306 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_77, output4_2, output_79, long4_2, batch_norm_36, output_82, output4_3, output4_add_4], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_42.run(buf305, buf303, primals_337, primals_338, primals_339, primals_340, primals_341, buf288, primals_322, buf271, buf291, primals_329, buf275, buf306, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_83], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_54], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf308, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_54], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf307, primals_344, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf309, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_54], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf307, primals_345, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf310, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_54], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf307, primals_346, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf311, (4, 16, 4, 4), (256, 16, 4, 1))
        buf313 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [p_18, d1_55], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf313, buf307, 1024, grid=grid(1024), stream=stream0)
        buf314 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [cat_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf313, buf309, buf310, buf311, buf314, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf315, (4, 4, 4, 4), (64, 16, 4, 1))
        buf316 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [cat_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf313, buf315, buf309, buf310, buf311, buf316, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_84], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf317, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf318 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf319 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [output_79, long4_2, batch_norm_37, output_85, output4_4, output4_add_5], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_43.run(buf319, buf317, primals_349, primals_350, primals_351, primals_352, primals_353, buf305, primals_329, buf275, buf318, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_86], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d1_57], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_355, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf321, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d2_57], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf320, primals_356, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf322, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d3_57], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf320, primals_357, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf323, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [d4_57], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf320, primals_358, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf324, (4, 16, 4, 4), (256, 16, 4, 1))
        buf326 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [p_19, d1_58], Original ATen: [aten.avg_pool2d, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_38.run(buf326, buf320, 1024, grid=grid(1024), stream=stream0)
        buf327 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [cat_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf326, buf322, buf323, buf324, buf327, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_170], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf328, (4, 4, 4, 4), (64, 16, 4, 1))
        buf329 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf326, buf328, buf322, buf323, buf324, buf329, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_87], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf330, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf331 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [output_85, output4_4, batch_norm_38, output_88, output4_5], Original ATen: [aten._prelu_kernel, aten.add, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_44.run(buf332, buf330, primals_361, primals_362, primals_363, primals_364, primals_365, buf318, primals_353, buf305, 4096, grid=grid(4096), stream=stream0)
        del buf305
        # Topologically Sorted Source Nodes: [conv2d_172], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf334 = buf333; del buf333  # reuse
        buf335 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [conv2d_172, up4_conv4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_45.run(buf334, primals_367, primals_368, primals_369, primals_370, primals_371, buf335, 4096, grid=grid(4096), stream=stream0)
        del primals_367
        buf336 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_46.run(buf336, 8, grid=grid(8), stream=stream0)
        buf337 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_47.run(buf337, 8, grid=grid(8), stream=stream0)
        buf338 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_46.run(buf338, 8, grid=grid(8), stream=stream0)
        buf339 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_47.run(buf339, 8, grid=grid(8), stream=stream0)
        buf340 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48.run(buf340, 8, grid=grid(8), stream=stream0)
        buf342 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [up4_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_48.run(buf342, 8, grid=grid(8), stream=stream0)
        buf341 = buf224; del buf224  # reuse
        buf344 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [up4, up4_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_49.run(buf344, buf336, buf338, buf335, primals_372, buf339, buf340, buf337, buf342, 16384, grid=grid(16384), stream=stream0)
        del buf335
        # Topologically Sorted Source Nodes: [output_89], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, primals_373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d1_60], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_374, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf346, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [d2_60], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf345, primals_375, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf347, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [conv2d_176], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf220, primals_380, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf348 = buf346; del buf346  # reuse
        buf350 = buf349; del buf349  # reuse
        buf351 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_90, output_91, conv2d_176, up3_conv3, add_218], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_50.run(buf348, buf350, buf347, primals_381, primals_376, primals_377, primals_378, primals_379, primals_382, primals_383, primals_384, primals_385, buf351, 8192, grid=grid(8192), stream=stream0)
        del buf347
        del primals_381
        buf352 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_51.run(buf352, 16, grid=grid(16), stream=stream0)
        buf353 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_52.run(buf353, 16, grid=grid(16), stream=stream0)
        buf354 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_51.run(buf354, 16, grid=grid(16), stream=stream0)
        buf355 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_52.run(buf355, 16, grid=grid(16), stream=stream0)
        buf356 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53.run(buf356, 16, grid=grid(16), stream=stream0)
        buf358 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [up3_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_53.run(buf358, 16, grid=grid(16), stream=stream0)
        buf357 = reinterpret_tensor(buf7, (4, 32, 16, 16), (8192, 256, 16, 1), 0); del buf7  # reuse
        buf360 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [up3, up3_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_54.run(buf360, buf352, buf354, buf351, primals_386, buf355, buf356, buf353, buf358, 32768, grid=grid(32768), stream=stream0)
        del buf351
        # Topologically Sorted Source Nodes: [output_92], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 24, 16, 16), (6144, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d1_61], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf362, (4, 24, 16, 16), (6144, 256, 16, 1))
        # Topologically Sorted Source Nodes: [d2_61], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf361, primals_389, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf363, (4, 24, 16, 16), (6144, 256, 16, 1))
        # Topologically Sorted Source Nodes: [conv2d_180], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf78, primals_394, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf364 = buf362; del buf362  # reuse
        buf366 = buf365; del buf365  # reuse
        buf367 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [output_93, output_94, conv2d_180, up2_conv2, add_220], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_55.run(buf364, buf366, buf363, primals_395, primals_390, primals_391, primals_392, primals_393, primals_396, primals_397, primals_398, primals_399, buf367, 24576, grid=grid(24576), stream=stream0)
        del buf363
        del primals_395
        buf368 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_56.run(buf368, 32, grid=grid(32), stream=stream0)
        buf369 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_57.run(buf369, 32, grid=grid(32), stream=stream0)
        buf370 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_56.run(buf370, 32, grid=grid(32), stream=stream0)
        buf371 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_57.run(buf371, 32, grid=grid(32), stream=stream0)
        buf372 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58.run(buf372, 32, grid=grid(32), stream=stream0)
        buf374 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_58.run(buf374, 32, grid=grid(32), stream=stream0)
        buf373 = empty_strided_cuda((4, 24, 32, 32), (24576, 1024, 32, 1), torch.float32)
        buf376 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [up2, up2_1], Original ATen: [aten._prelu_kernel, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_59.run(buf376, buf368, buf370, buf367, primals_400, buf371, buf372, buf369, buf374, 98304, grid=grid(98304), stream=stream0)
        del buf367
        # Topologically Sorted Source Nodes: [output_95], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [d1_62], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf378, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [d2_62], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf377, primals_403, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf379, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_184], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf14, primals_408, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf380 = buf378; del buf378  # reuse
        buf382 = buf381; del buf381  # reuse
        buf383 = buf4; del buf4  # reuse
        buf384 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [output_96, output_97, conv2d_184, up1_conv1, add_222, up1], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_add_convolution_60.run(buf380, buf382, buf384, buf379, primals_409, primals_404, primals_405, primals_406, primals_407, primals_410, primals_411, primals_412, primals_413, primals_414, 32768, grid=grid(32768), stream=stream0)
        del primals_409
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 2, 32, 32), (2048, 1024, 32, 1))
        buf386 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf386, 64, grid=grid(64), stream=stream0)
        buf387 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_62.run(buf387, 64, grid=grid(64), stream=stream0)
        buf388 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf388, 64, grid=grid(64), stream=stream0)
        buf389 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_62.run(buf389, 64, grid=grid(64), stream=stream0)
        buf390 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63.run(buf390, 64, grid=grid(64), stream=stream0)
        buf392 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_63.run(buf392, 64, grid=grid(64), stream=stream0)
        buf391 = reinterpret_tensor(buf379, (4, 2, 64, 64), (8192, 4096, 64, 1), 0); del buf379  # reuse
        buf394 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [input_8, pred1], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sub_64.run(buf394, buf386, buf388, buf385, primals_416, buf389, buf390, buf387, buf392, 32768, grid=grid(32768), stream=stream0)
        del buf385
        del primals_416
    return (buf394, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, buf0, buf1, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf16, buf18, buf19, buf20, buf22, buf23, buf25, buf26, buf27, buf29, buf30, buf31, buf32, buf33, buf35, buf36, buf38, buf39, buf40, buf42, buf43, buf44, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf56, buf58, buf59, buf60, buf61, buf62, buf65, buf66, buf68, buf69, buf70, buf72, buf73, buf74, buf75, buf76, buf78, buf79, buf80, buf82, buf83, buf84, buf86, buf87, buf89, buf90, buf91, buf93, buf94, buf95, buf96, buf97, buf99, buf100, buf102, buf103, buf104, buf106, buf107, buf108, buf109, buf110, buf112, buf113, buf115, buf116, buf118, buf119, buf120, buf122, buf123, buf124, buf125, buf126, buf129, buf130, buf133, buf134, buf136, buf137, buf138, buf140, buf141, buf142, buf143, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf154, buf156, buf157, buf158, buf159, buf160, buf163, buf164, buf167, buf168, buf170, buf171, buf172, buf174, buf175, buf176, buf177, buf178, buf180, buf181, buf183, buf184, buf185, buf187, buf188, buf189, buf190, buf191, buf194, buf195, buf197, buf198, buf199, buf201, buf202, buf203, buf204, buf205, buf207, buf208, buf210, buf211, buf212, buf214, buf215, buf216, buf217, buf218, buf220, buf222, buf223, buf225, buf226, buf227, buf229, buf230, buf232, buf233, buf234, buf236, buf237, buf238, buf239, buf240, buf242, buf243, buf245, buf246, buf247, buf249, buf250, buf251, buf252, buf253, buf255, buf256, buf258, buf259, buf261, buf262, buf263, buf265, buf266, buf267, buf268, buf269, buf272, buf273, buf276, buf277, buf279, buf280, buf281, buf283, buf284, buf285, buf286, buf287, buf289, buf290, buf292, buf293, buf295, buf296, buf297, buf299, buf300, buf301, buf302, buf303, buf306, buf307, buf309, buf310, buf311, buf313, buf314, buf315, buf316, buf317, buf319, buf320, buf322, buf323, buf324, buf326, buf327, buf328, buf329, buf330, buf332, buf334, buf336, buf337, buf338, buf339, buf340, buf342, buf344, buf345, buf348, buf350, buf352, buf353, buf354, buf355, buf356, buf358, buf360, buf361, buf364, buf366, buf368, buf369, buf370, buf371, buf372, buf374, buf376, buf377, buf380, buf382, buf384, buf386, buf387, buf388, buf389, buf390, buf392, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((8, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((6, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((24, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((6, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((24, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((6, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((24, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((6, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((6, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((24, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((48, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((2, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
