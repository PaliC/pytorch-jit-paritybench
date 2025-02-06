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


# kernel path: inductor_cache/ej/cejvytsx6xwsvzohb5etxhynnfjhai565cgso237bjfdgk7e4o2u.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1 => convolution
# Graph fragment:
#   %convolution : [num_users=5] = call_function[target=torch.ops.aten.convolution.default](args = (%view, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctx2t37l5irgr3xfsxi2rnba62ipmddtjmrwhec3yhfv7nhdws2a.py
# Topologically Sorted Source Nodes: [y, group_norm, y_1, y_2, y_3, group_norm_1, y_4, y_5, y_6, group_norm_2, y_7, y_8, y_9, group_norm_3, y_10, y_11], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
# Source node to ATen node mapping:
#   group_norm => add, add_1, mul_1, rsqrt, var_mean
#   group_norm_1 => add_3, add_4, mul_4, rsqrt_1, var_mean_1
#   group_norm_2 => add_6, add_7, mul_7, rsqrt_2, var_mean_2
#   group_norm_3 => add_10, add_9, mul_10, rsqrt_3, var_mean_3
#   y => convolution_1
#   y_1 => gt, mul_2, where
#   y_10 => gt_3, mul_11, where_3
#   y_11 => add_11
#   y_2 => add_2
#   y_3 => convolution_2
#   y_4 => gt_1, mul_5, where_1
#   y_5 => add_5
#   y_6 => convolution_3
#   y_7 => gt_2, mul_8, where_2
#   y_8 => add_8
#   y_9 => convolution_4
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, 0), kwargs = {})
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_8, %primals_9, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_3, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %unsqueeze_11), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %unsqueeze_8), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_4, %mul_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %where_1), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_12, %primals_13, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_17), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_14), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.2), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_7, %mul_8), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %where_2), kwargs = {})
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_16, %primals_17, [1, 1], [5, 5], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_7, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %unsqueeze_23), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_20), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.2), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_10, %mul_11), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %where_3), kwargs = {})
triton_per_fused_add_convolution_leaky_relu_native_group_norm_1 = async_compile.triton('triton_per_fused_add_convolution_leaky_relu_native_group_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_leaky_relu_native_group_norm_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 16, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_leaky_relu_native_group_norm_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex
    r2 = rindex // 16
    tmp0 = tl.load(in_out_ptr0 + (r3 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr2 + (r3 + 64*x0), xmask, other=0.0)
    tmp25 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr4 + (r3 + 64*x0), xmask, other=0.0)
    tmp45 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_out_ptr6 + (r3 + 64*x0), xmask, other=0.0)
    tmp65 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr8 + (r2), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (r2), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr10 + (r2), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr11 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = tmp33 / tmp11
    tmp35 = tmp27 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None]
    tmp41 = tmp40 / tmp19
    tmp42 = tmp41 + tmp21
    tmp43 = libdevice.rsqrt(tmp42)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(xmask, tmp47, 0)
    tmp50 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = tmp53 / tmp11
    tmp55 = tmp47 - tmp54
    tmp56 = tmp55 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.where(xmask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None]
    tmp61 = tmp60 / tmp19
    tmp62 = tmp61 + tmp21
    tmp63 = libdevice.rsqrt(tmp62)
    tmp66 = tmp64 + tmp65
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
    tmp69 = tl.where(xmask, tmp67, 0)
    tmp70 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp72 = tl.where(xmask, tmp70, 0)
    tmp73 = tl.sum(tmp72, 1)[:, None]
    tmp74 = tmp73 / tmp11
    tmp75 = tmp67 - tmp74
    tmp76 = tmp75 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.where(xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tmp81 = tmp80 / tmp19
    tmp82 = tmp81 + tmp21
    tmp83 = libdevice.rsqrt(tmp82)
    tmp84 = tmp2 - tmp12
    tmp85 = tmp84 * tmp23
    tmp87 = tmp85 * tmp86
    tmp89 = tmp87 + tmp88
    tmp90 = tmp46 - tmp54
    tmp91 = tmp90 * tmp63
    tmp93 = tmp91 * tmp92
    tmp95 = tmp93 + tmp94
    tmp96 = tmp66 - tmp74
    tmp97 = tmp96 * tmp83
    tmp99 = tmp97 * tmp98
    tmp101 = tmp99 + tmp100
    tmp102 = tmp26 - tmp34
    tmp103 = tmp102 * tmp43
    tmp105 = tmp103 * tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = 0.0
    tmp109 = tmp89 > tmp108
    tmp110 = 0.2
    tmp111 = tmp89 * tmp110
    tmp112 = tl.where(tmp109, tmp89, tmp111)
    tmp113 = tmp112 + tmp108
    tmp114 = tmp95 > tmp108
    tmp115 = tmp95 * tmp110
    tmp116 = tl.where(tmp114, tmp95, tmp115)
    tmp117 = tmp113 + tmp116
    tmp118 = tmp101 > tmp108
    tmp119 = tmp101 * tmp110
    tmp120 = tl.where(tmp118, tmp101, tmp119)
    tmp121 = tmp117 + tmp120
    tmp122 = tmp107 > tmp108
    tmp123 = tmp107 * tmp110
    tmp124 = tl.where(tmp122, tmp107, tmp123)
    tmp125 = tmp121 + tmp124
    tl.store(in_out_ptr0 + (r3 + 64*x0), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(in_out_ptr2 + (r3 + 64*x0), tmp26, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr3 + (x0), tmp43, xmask)
    tl.store(in_out_ptr4 + (r3 + 64*x0), tmp46, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr5 + (x0), tmp63, xmask)
    tl.store(in_out_ptr6 + (r3 + 64*x0), tmp66, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr7 + (x0), tmp83, xmask)
    tl.store(in_out_ptr8 + (r3 + 64*x0), tmp125, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp34, xmask)
    tl.store(out_ptr2 + (x0), tmp54, xmask)
    tl.store(out_ptr3 + (x0), tmp74, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxu4pfcv4yn6gvzws3ssh6rdyqjgqqadoynvdfgnpsf4vw55u2m.py
# Topologically Sorted Source Nodes: [y_48, group_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_16 => add_48, add_49, mul_49, rsqrt_16, var_mean_16
#   y_48 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_20, %primals_76, %primals_77, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_33, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_16 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_34, %unsqueeze_101), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_98), kwargs = {})
triton_per_fused_convolution_native_group_norm_2 = async_compile.triton('triton_per_fused_convolution_native_group_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex
    r2 = rindex // 16
    tmp0 = tl.load(in_out_ptr0 + (r3 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp2 - tmp12
    tmp25 = tmp24 * tmp23
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r3 + 64*x0), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(out_ptr1 + (r3 + 64*x0), tmp29, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4hdglrik46anasuvweqbx7dsoj5fp7fhptid2su4hp2tzch6g7m.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_59, %add_35], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 8)
    x0 = (xindex % 16)
    x2 = xindex // 128
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tmp10 + tmp6
    tmp12 = tl.load(in_ptr1 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp13 = tmp12 > tmp6
    tmp14 = tmp12 * tmp8
    tmp15 = tl.where(tmp13, tmp12, tmp14)
    tmp16 = tmp11 + tmp15
    tmp17 = tl.load(in_ptr2 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp18 = tmp17 > tmp6
    tmp19 = tmp17 * tmp8
    tmp20 = tl.where(tmp18, tmp17, tmp19)
    tmp21 = tmp16 + tmp20
    tmp22 = tl.load(in_ptr3 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp23 = tmp22 > tmp6
    tmp24 = tmp22 * tmp8
    tmp25 = tl.where(tmp23, tmp22, tmp24)
    tmp26 = tmp21 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 8, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + (x0 + 16*((-4) + x1) + 64*x2), tmp29 & xmask, other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x3), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 4, 4, 4), (64, 64, 16, 4, 1))
    assert_size_stride(primals_2, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (2, ), (1, ))
    assert_size_stride(primals_4, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (2, ), (1, ))
    assert_size_stride(primals_22, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (2, ), (1, ))
    assert_size_stride(primals_40, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, ), (1, ))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, ), (1, ))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_57, (2, ), (1, ))
    assert_size_stride(primals_58, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (4, ), (1, ))
    assert_size_stride(primals_61, (4, ), (1, ))
    assert_size_stride(primals_62, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_67, (4, ), (1, ))
    assert_size_stride(primals_68, (4, ), (1, ))
    assert_size_stride(primals_69, (4, ), (1, ))
    assert_size_stride(primals_70, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (4, ), (1, ))
    assert_size_stride(primals_73, (4, ), (1, ))
    assert_size_stride(primals_74, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_75, (2, ), (1, ))
    assert_size_stride(primals_76, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_77, (4, ), (1, ))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_81, (4, ), (1, ))
    assert_size_stride(primals_82, (4, ), (1, ))
    assert_size_stride(primals_83, (4, ), (1, ))
    assert_size_stride(primals_84, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (4, ), (1, ))
    assert_size_stride(primals_87, (4, ), (1, ))
    assert_size_stride(primals_88, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_89, (4, ), (1, ))
    assert_size_stride(primals_90, (4, ), (1, ))
    assert_size_stride(primals_91, (4, ), (1, ))
    assert_size_stride(primals_92, (2, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_93, (2, ), (1, ))
    assert_size_stride(primals_94, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (4, ), (1, ))
    assert_size_stride(primals_97, (4, ), (1, ))
    assert_size_stride(primals_98, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_99, (4, ), (1, ))
    assert_size_stride(primals_100, (4, ), (1, ))
    assert_size_stride(primals_101, (4, ), (1, ))
    assert_size_stride(primals_102, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_103, (4, ), (1, ))
    assert_size_stride(primals_104, (4, ), (1, ))
    assert_size_stride(primals_105, (4, ), (1, ))
    assert_size_stride(primals_106, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_107, (4, ), (1, ))
    assert_size_stride(primals_108, (4, ), (1, ))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (2, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_111, (2, ), (1, ))
    assert_size_stride(primals_112, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (4, ), (1, ))
    assert_size_stride(primals_115, (4, ), (1, ))
    assert_size_stride(primals_116, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_117, (4, ), (1, ))
    assert_size_stride(primals_118, (4, ), (1, ))
    assert_size_stride(primals_119, (4, ), (1, ))
    assert_size_stride(primals_120, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_121, (4, ), (1, ))
    assert_size_stride(primals_122, (4, ), (1, ))
    assert_size_stride(primals_123, (4, ), (1, ))
    assert_size_stride(primals_124, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (4, ), (1, ))
    assert_size_stride(primals_127, (4, ), (1, ))
    assert_size_stride(primals_128, (2, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_129, (2, ), (1, ))
    assert_size_stride(primals_130, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_131, (4, ), (1, ))
    assert_size_stride(primals_132, (4, ), (1, ))
    assert_size_stride(primals_133, (4, ), (1, ))
    assert_size_stride(primals_134, (4, 2, 5, 5), (50, 25, 5, 1))
    assert_size_stride(primals_135, (4, ), (1, ))
    assert_size_stride(primals_136, (4, ), (1, ))
    assert_size_stride(primals_137, (4, ), (1, ))
    assert_size_stride(primals_138, (4, 2, 7, 7), (98, 49, 7, 1))
    assert_size_stride(primals_139, (4, ), (1, ))
    assert_size_stride(primals_140, (4, ), (1, ))
    assert_size_stride(primals_141, (4, ), (1, ))
    assert_size_stride(primals_142, (4, 2, 11, 11), (242, 121, 11, 1))
    assert_size_stride(primals_143, (4, ), (1, ))
    assert_size_stride(primals_144, (4, ), (1, ))
    assert_size_stride(primals_145, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4), (32, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_3, 128, grid=grid(128), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf1, primals_12, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf1, primals_16, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf1, primals_8, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 4, 4, 4), (64, 16, 4, 1))
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf7 = reinterpret_tensor(buf5, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf5  # reuse
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf26 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf28 = reinterpret_tensor(buf26, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf26  # reuse
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf14 = reinterpret_tensor(buf12, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf12  # reuse
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf19 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf21 = reinterpret_tensor(buf19, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf19  # reuse
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf30 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [y, group_norm, y_1, y_2, y_3, group_norm_1, y_4, y_5, y_6, group_norm_2, y_7, y_8, y_9, group_norm_3, y_10, y_11], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_leaky_relu_native_group_norm_1.run(buf3, buf7, buf24, buf28, buf10, buf14, buf17, buf21, buf30, primals_5, primals_17, primals_9, primals_13, primals_6, primals_7, primals_10, primals_11, primals_14, primals_15, primals_18, primals_19, buf4, buf25, buf11, buf18, 4, 64, grid=grid(4), stream=stream0)
        del primals_13
        del primals_17
        del primals_5
        del primals_9
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 2, 4, 4), (32, 16, 4, 1))
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf32, primals_21, 128, grid=grid(128), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf32, primals_26, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf32, primals_30, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_21], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf32, primals_34, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 4, 4, 4), (64, 16, 4, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf36 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf38 = reinterpret_tensor(buf36, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf36  # reuse
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf59 = reinterpret_tensor(buf57, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf57  # reuse
        buf41 = buf40; del buf40  # reuse
        buf42 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf43 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf45 = reinterpret_tensor(buf43, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf43  # reuse
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf50 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf52 = reinterpret_tensor(buf50, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf50  # reuse
        buf39 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf61 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [y_12, group_norm_4, y_13, y_14, y_15, group_norm_5, y_16, y_17, y_18, group_norm_6, y_19, y_20, y_21, group_norm_7, y_22, y_23], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_leaky_relu_native_group_norm_1.run(buf34, buf38, buf55, buf59, buf41, buf45, buf48, buf52, buf61, primals_23, primals_35, primals_27, primals_31, primals_24, primals_25, primals_28, primals_29, primals_32, primals_33, primals_36, primals_37, buf35, buf56, buf42, buf49, 4, 64, grid=grid(4), stream=stream0)
        del primals_23
        del primals_27
        del primals_31
        del primals_35
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 2, 4, 4), (32, 16, 4, 1))
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf63, primals_39, 128, grid=grid(128), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_27], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf63, primals_44, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_30], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf63, primals_48, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_33], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf63, primals_52, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 4, 4, 4), (64, 16, 4, 1))
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf67 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf69 = reinterpret_tensor(buf67, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf67  # reuse
        buf86 = buf85; del buf85  # reuse
        buf87 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf88 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf90 = reinterpret_tensor(buf88, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf88  # reuse
        buf72 = buf71; del buf71  # reuse
        buf73 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf74 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf76 = reinterpret_tensor(buf74, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf74  # reuse
        buf79 = buf78; del buf78  # reuse
        buf80 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf81 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf83 = reinterpret_tensor(buf81, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf81  # reuse
        buf70 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf92 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [y_24, group_norm_8, y_25, y_26, y_27, group_norm_9, y_28, y_29, y_30, group_norm_10, y_31, y_32, y_33, group_norm_11, y_34, y_35], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_leaky_relu_native_group_norm_1.run(buf65, buf69, buf86, buf90, buf72, buf76, buf79, buf83, buf92, primals_41, primals_53, primals_45, primals_49, primals_42, primals_43, primals_46, primals_47, primals_50, primals_51, primals_54, primals_55, buf66, buf87, buf73, buf80, 4, 64, grid=grid(4), stream=stream0)
        del primals_41
        del primals_45
        del primals_49
        del primals_53
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 2, 4, 4), (32, 16, 4, 1))
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf94, primals_57, 128, grid=grid(128), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [y_36], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_39], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf94, primals_62, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_42], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf94, primals_66, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_45], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf94, primals_70, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 4, 4, 4), (64, 16, 4, 1))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf98 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf100 = reinterpret_tensor(buf98, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf98  # reuse
        buf117 = buf116; del buf116  # reuse
        buf118 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf119 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf121 = reinterpret_tensor(buf119, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf119  # reuse
        buf103 = buf102; del buf102  # reuse
        buf104 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf105 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf107 = reinterpret_tensor(buf105, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf105  # reuse
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf112 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf114 = reinterpret_tensor(buf112, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf112  # reuse
        buf101 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf123 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [y_36, group_norm_12, y_37, y_38, y_39, group_norm_13, y_40, y_41, y_42, group_norm_14, y_43, y_44, y_45, group_norm_15, y_46, y_47], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_leaky_relu_native_group_norm_1.run(buf96, buf100, buf117, buf121, buf103, buf107, buf110, buf114, buf123, primals_59, primals_71, primals_63, primals_67, primals_60, primals_61, primals_64, primals_65, primals_68, primals_69, primals_72, primals_73, buf97, buf118, buf104, buf111, 4, 64, grid=grid(4), stream=stream0)
        del primals_59
        del primals_63
        del primals_67
        del primals_71
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 2, 4, 4), (32, 16, 4, 1))
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf125, primals_75, 128, grid=grid(128), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [y_48], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 4, 4, 4), (64, 16, 4, 1))
        buf127 = buf126; del buf126  # reuse
        buf128 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf129 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf131 = reinterpret_tensor(buf129, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf129  # reuse
        buf132 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_48, group_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf127, buf131, primals_77, primals_78, primals_79, buf128, buf132, 4, 64, grid=grid(4), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [y_51], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf125, primals_80, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 4, 4, 4), (64, 16, 4, 1))
        buf134 = buf133; del buf133  # reuse
        buf135 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf136 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf138 = reinterpret_tensor(buf136, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf136  # reuse
        buf139 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_51, group_norm_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf134, buf138, primals_81, primals_82, primals_83, buf135, buf139, 4, 64, grid=grid(4), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [y_54], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf125, primals_84, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 4, 4, 4), (64, 16, 4, 1))
        buf141 = buf140; del buf140  # reuse
        buf142 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf143 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf145 = reinterpret_tensor(buf143, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf143  # reuse
        buf146 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_54, group_norm_18], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf141, buf145, primals_85, primals_86, primals_87, buf142, buf146, 4, 64, grid=grid(4), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [y_57], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf125, primals_88, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 4, 4, 4), (64, 16, 4, 1))
        buf148 = buf147; del buf147  # reuse
        buf149 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf150 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf152 = reinterpret_tensor(buf150, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf150  # reuse
        buf153 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_57, group_norm_19], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf148, buf152, primals_89, primals_90, primals_91, buf149, buf153, 4, 64, grid=grid(4), stream=stream0)
        del primals_89
        buf154 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf132, buf139, buf146, buf153, buf92, buf154, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 2, 4, 4), (32, 16, 4, 1))
        buf156 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf156, primals_93, 128, grid=grid(128), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [y_60], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 4, 4, 4), (64, 16, 4, 1))
        buf158 = buf157; del buf157  # reuse
        buf159 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf160 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf162 = reinterpret_tensor(buf160, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf160  # reuse
        buf163 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [y_60, group_norm_20], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf158, buf162, primals_95, primals_96, primals_97, buf159, buf163, 4, 64, grid=grid(4), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [y_63], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf156, primals_98, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 4, 4, 4), (64, 16, 4, 1))
        buf165 = buf164; del buf164  # reuse
        buf166 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf167 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf169 = reinterpret_tensor(buf167, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf167  # reuse
        buf170 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [y_63, group_norm_21], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf165, buf169, primals_99, primals_100, primals_101, buf166, buf170, 4, 64, grid=grid(4), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [y_66], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf156, primals_102, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 4, 4, 4), (64, 16, 4, 1))
        buf172 = buf171; del buf171  # reuse
        buf173 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf174 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf176 = reinterpret_tensor(buf174, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf174  # reuse
        buf177 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [y_66, group_norm_22], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf172, buf176, primals_103, primals_104, primals_105, buf173, buf177, 4, 64, grid=grid(4), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [y_69], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf156, primals_106, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 4, 4, 4), (64, 16, 4, 1))
        buf179 = buf178; del buf178  # reuse
        buf180 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf181 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf183 = reinterpret_tensor(buf181, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf181  # reuse
        buf184 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [y_69, group_norm_23], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf179, buf183, primals_107, primals_108, primals_109, buf180, buf184, 4, 64, grid=grid(4), stream=stream0)
        del primals_107
        buf185 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf163, buf170, buf177, buf184, buf61, buf185, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 2, 4, 4), (32, 16, 4, 1))
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf187, primals_111, 128, grid=grid(128), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [y_72], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 4, 4, 4), (64, 16, 4, 1))
        buf189 = buf188; del buf188  # reuse
        buf190 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf191 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf193 = reinterpret_tensor(buf191, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf191  # reuse
        buf194 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [y_72, group_norm_24], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf189, buf193, primals_113, primals_114, primals_115, buf190, buf194, 4, 64, grid=grid(4), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [y_75], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf187, primals_116, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 4, 4, 4), (64, 16, 4, 1))
        buf196 = buf195; del buf195  # reuse
        buf197 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf198 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf200 = reinterpret_tensor(buf198, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf198  # reuse
        buf201 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [y_75, group_norm_25], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf196, buf200, primals_117, primals_118, primals_119, buf197, buf201, 4, 64, grid=grid(4), stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [y_78], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf187, primals_120, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 4, 4, 4), (64, 16, 4, 1))
        buf203 = buf202; del buf202  # reuse
        buf204 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf205 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf207 = reinterpret_tensor(buf205, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf205  # reuse
        buf208 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [y_78, group_norm_26], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf203, buf207, primals_121, primals_122, primals_123, buf204, buf208, 4, 64, grid=grid(4), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [y_81], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf187, primals_124, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 4, 4, 4), (64, 16, 4, 1))
        buf210 = buf209; del buf209  # reuse
        buf211 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf212 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf214 = reinterpret_tensor(buf212, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf212  # reuse
        buf215 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [y_81, group_norm_27], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_2.run(buf210, buf214, primals_125, primals_126, primals_127, buf211, buf215, 4, 64, grid=grid(4), stream=stream0)
        del primals_125
        buf216 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf194, buf201, buf208, buf215, buf30, buf216, 512, grid=grid(512), stream=stream0)
        del buf194
        del buf201
        del buf208
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 2, 4, 4), (32, 16, 4, 1))
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf218, primals_129, 128, grid=grid(128), stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [y_84], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_87], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf218, primals_134, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_90], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf218, primals_138, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [y_93], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf218, primals_142, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 4, 4, 4), (64, 16, 4, 1))
        buf220 = buf219; del buf219  # reuse
        buf221 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf222 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf224 = reinterpret_tensor(buf222, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf222  # reuse
        buf241 = buf240; del buf240  # reuse
        buf242 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf243 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf245 = reinterpret_tensor(buf243, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf243  # reuse
        buf227 = buf226; del buf226  # reuse
        buf228 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf229 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf231 = reinterpret_tensor(buf229, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf229  # reuse
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf236 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf238 = reinterpret_tensor(buf236, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf236  # reuse
        buf225 = buf215; del buf215  # reuse
        buf247 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [y_84, group_norm_28, y_85, y_86, y_87, group_norm_29, y_88, y_89, y_90, group_norm_30, y_91, y_92, y_93, group_norm_31, y_94, y_95], Original ATen: [aten.convolution, aten.native_group_norm, aten.leaky_relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_leaky_relu_native_group_norm_1.run(buf220, buf224, buf241, buf245, buf227, buf231, buf234, buf238, buf247, primals_131, primals_143, primals_135, primals_139, primals_132, primals_133, primals_136, primals_137, primals_140, primals_141, primals_144, primals_145, buf221, buf242, buf228, buf235, 4, 64, grid=grid(4), stream=stream0)
        del primals_131
        del primals_135
        del primals_139
        del primals_143
    return (reinterpret_tensor(buf247, (4, 1, 4, 4, 4), (64, 64, 16, 4, 1), 0), primals_2, primals_4, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_22, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_40, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_140, primals_141, primals_142, primals_144, primals_145, reinterpret_tensor(primals_1, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf1, buf3, buf4, buf7, buf10, buf11, buf14, buf17, buf18, buf21, buf24, buf25, buf28, buf30, buf32, buf34, buf35, buf38, buf41, buf42, buf45, buf48, buf49, buf52, buf55, buf56, buf59, buf61, buf63, buf65, buf66, buf69, buf72, buf73, buf76, buf79, buf80, buf83, buf86, buf87, buf90, buf92, buf94, buf96, buf97, buf100, buf103, buf104, buf107, buf110, buf111, buf114, buf117, buf118, buf121, buf123, buf125, buf127, buf128, buf131, buf134, buf135, buf138, buf141, buf142, buf145, buf148, buf149, buf152, buf154, buf156, buf158, buf159, buf162, buf165, buf166, buf169, buf172, buf173, buf176, buf179, buf180, buf183, buf185, buf187, buf189, buf190, buf193, buf196, buf197, buf200, buf203, buf204, buf207, buf210, buf211, buf214, buf216, buf218, buf220, buf221, buf224, buf227, buf228, buf231, buf234, buf235, buf238, buf241, buf242, buf245, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 4, 4, 4), (64, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((2, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((4, 2, 5, 5), (50, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((4, 2, 7, 7), (98, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((4, 2, 11, 11), (242, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
