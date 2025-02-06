# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/dm/cdmd6wb46xuhbuif3i7sqck3sraxssnpwfcfenua2gbqplbjykpt.py
# Topologically Sorted Source Nodes: [sub, sub_1, gt_area, sub_2, sub_3, pr_area, add_1, inter, union, gt_cent_x, pr_cent_x, sub_6, pow_1, gt_cent_y, pr_cent_y, sub_7, pow_2, cent_dis, lt_1, rb_1, sub_8, pow_3, diag_dis, add_3, reg, sub_15, sub_16, gt_area_1, sub_17, sub_18, pr_area_1, add_5, inter_1, union_1, gt_w, gt_h, truediv_2, atan, pr_w, pr_h, truediv_3, atan_1, atan_diff, v, v_1], Original ATen: [aten.sub, aten.mul, aten.add, aten.mean, aten.pow, aten.minimum, aten.maximum, aten.sum, aten.div, aten.atan]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_3 => add_3
#   add_5 => add_5
#   atan => atan
#   atan_1 => atan_1
#   atan_diff => sub_14
#   cent_dis => add_2
#   diag_dis => sum_1
#   gt_area => mul
#   gt_area_1 => mul_4
#   gt_cent_x => mean
#   gt_cent_y => mean_1
#   gt_h => sub_11
#   gt_w => sub_10
#   inter => mul_2
#   inter_1 => mul_6
#   lt_1 => minimum_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pr_area => mul_1
#   pr_area_1 => mul_5
#   pr_cent_x => mean_2
#   pr_cent_y => mean_3
#   pr_h => sub_13
#   pr_w => sub_12
#   rb_1 => maximum_1
#   reg => div_1
#   sub => sub
#   sub_1 => sub_1
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sub_18 => sub_18
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   truediv_2 => div_2
#   truediv_3 => div_3
#   union => sub_5
#   union_1 => sub_20
#   v => pow_4
#   v_1 => mul_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %select_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %sub_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mul_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_20, [-1]), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_24, [-1]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mean_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2.0), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_22, [-1]), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_26, [-1]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_1, %mean_3), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2.0), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%slice_28, %slice_30), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%slice_32, %slice_34), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %maximum_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1]), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-05), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2, %add_3), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_18, %select_19), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_20, %select_21), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %sub_16), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_22, %select_23), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_24, %select_25), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %sub_18), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_26, %select_27), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %mul_6), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_10, %select_11), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_12, %select_13), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_10, %sub_11), kwargs = {})
#   %atan : [num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%div_2,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_14, %select_15), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_16, %select_17), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_12, %sub_13), kwargs = {})
#   %atan_1 : [num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%div_3,), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%atan, %atan_1), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_14, 2.0), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, 0.4052847345693511), kwargs = {})
triton_poi_fused_add_atan_div_maximum_mean_minimum_mul_pow_sub_sum_0 = async_compile.triton('triton_poi_fused_add_atan_div_maximum_mean_minimum_mul_pow_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_atan_div_maximum_mean_minimum_mul_pow_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_atan_div_maximum_mean_minimum_mul_pow_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp13 - tmp16
    tmp18 = tmp17 + tmp7
    tmp19 = triton_helpers.maximum(tmp18, tmp9)
    tmp20 = tmp10 * tmp19
    tmp21 = tmp0 - tmp3
    tmp22 = tmp11 - tmp14
    tmp23 = tmp21 * tmp22
    tmp24 = tmp1 - tmp4
    tmp25 = tmp12 - tmp15
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp20
    tmp29 = tmp3 + tmp0
    tmp30 = 2.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp4 + tmp1
    tmp33 = tmp32 / tmp30
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tmp14 + tmp11
    tmp37 = tmp36 / tmp30
    tmp38 = tmp15 + tmp12
    tmp39 = tmp38 / tmp30
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = triton_helpers.minimum(tmp3, tmp4)
    tmp44 = triton_helpers.maximum(tmp0, tmp1)
    tmp45 = tmp43 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = triton_helpers.minimum(tmp14, tmp15)
    tmp48 = triton_helpers.maximum(tmp11, tmp12)
    tmp49 = tmp47 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tmp46 + tmp50
    tmp52 = tmp51 + tmp7
    tmp53 = tmp42 / tmp52
    tmp54 = tmp21 / tmp22
    tmp55 = libdevice.atan(tmp54)
    tmp56 = tmp24 / tmp25
    tmp57 = libdevice.atan(tmp56)
    tmp58 = tmp55 - tmp57
    tmp59 = tmp58 * tmp58
    tmp60 = 0.4052847345693511
    tmp61 = tmp59 * tmp60
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp53, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr5 + (x0), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ap/capzqtaujn7bksdffyyioqvjiudvhvj7xmkdwjkjfsn6ddcx7mrr.py
# Topologically Sorted Source Nodes: [iou, diou, iou_1, sub_21, add_6, alpha, reg_1, ciou, loss, loss_1], Original ATen: [aten.div, aten.sub, aten.rsub, aten.add, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_6 => add_6
#   alpha => div_5
#   ciou => sub_22
#   diou => sub_9
#   iou => div
#   iou_1 => div_4
#   loss => sub_23
#   loss_1 => sum_2
#   reg_1 => mul_7
#   sub_21 => sub_21
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %sub_5), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, %sub_20), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_4), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_21, %mul_3), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, %add_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %mul_3), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_7), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sub_22), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_23,), kwargs = {})
triton_poi_fused_add_div_mul_rsub_sub_sum_1 = async_compile.triton('triton_poi_fused_add_div_mul_rsub_sub_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_rsub_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_rsub_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr5 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr2 + (1))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp30 = tl.load(in_ptr3 + (1))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp32 = tl.load(in_ptr4 + (1))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr5 + (1))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp44 = tl.load(in_ptr0 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp46 = tl.load(in_ptr1 + (2))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp49 = tl.load(in_ptr2 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr3 + (2))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp54 = tl.load(in_ptr4 + (2))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp56 = tl.load(in_ptr5 + (2))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp66 = tl.load(in_ptr0 + (3))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (3))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp71 = tl.load(in_ptr2 + (3))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp74 = tl.load(in_ptr3 + (3))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp76 = tl.load(in_ptr4 + (3))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp78 = tl.load(in_ptr5 + (3))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp4 = tmp1 / tmp3
    tmp7 = tmp4 - tmp6
    tmp14 = tmp11 / tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp16 + tmp9
    tmp18 = tmp9 / tmp17
    tmp19 = tmp18 * tmp9
    tmp20 = tmp7 - tmp19
    tmp21 = tmp15 - tmp20
    tmp26 = tmp23 / tmp25
    tmp29 = tmp26 - tmp28
    tmp36 = tmp33 / tmp35
    tmp37 = tmp15 - tmp36
    tmp38 = tmp37 + tmp31
    tmp39 = tmp31 / tmp38
    tmp40 = tmp39 * tmp31
    tmp41 = tmp29 - tmp40
    tmp42 = tmp15 - tmp41
    tmp43 = tmp21 + tmp42
    tmp48 = tmp45 / tmp47
    tmp51 = tmp48 - tmp50
    tmp58 = tmp55 / tmp57
    tmp59 = tmp15 - tmp58
    tmp60 = tmp59 + tmp53
    tmp61 = tmp53 / tmp60
    tmp62 = tmp61 * tmp53
    tmp63 = tmp51 - tmp62
    tmp64 = tmp15 - tmp63
    tmp65 = tmp43 + tmp64
    tmp70 = tmp67 / tmp69
    tmp73 = tmp70 - tmp72
    tmp80 = tmp77 / tmp79
    tmp81 = tmp15 - tmp80
    tmp82 = tmp81 + tmp75
    tmp83 = tmp75 / tmp82
    tmp84 = tmp83 * tmp75
    tmp85 = tmp73 - tmp84
    tmp86 = tmp15 - tmp85
    tmp87 = tmp65 + tmp86
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp87, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf5 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub, sub_1, gt_area, sub_2, sub_3, pr_area, add_1, inter, union, gt_cent_x, pr_cent_x, sub_6, pow_1, gt_cent_y, pr_cent_y, sub_7, pow_2, cent_dis, lt_1, rb_1, sub_8, pow_3, diag_dis, add_3, reg, sub_15, sub_16, gt_area_1, sub_17, sub_18, pr_area_1, add_5, inter_1, union_1, gt_w, gt_h, truediv_2, atan, pr_w, pr_h, truediv_3, atan_1, atan_diff, v, v_1], Original ATen: [aten.sub, aten.mul, aten.add, aten.mean, aten.pow, aten.minimum, aten.maximum, aten.sum, aten.div, aten.atan]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_atan_div_maximum_mean_minimum_mul_pow_sub_sum_0.run(arg0_1, arg1_1, buf0, buf1, buf2, buf3, buf4, buf5, 4, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        buf6 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [iou, diou, iou_1, sub_21, add_6, alpha, reg_1, ciou, loss, loss_1], Original ATen: [aten.div, aten.sub, aten.rsub, aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_rsub_sub_sum_1.run(buf0, buf1, buf2, buf5, buf3, buf4, buf6, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
        del buf2
        del buf3
        del buf4
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
