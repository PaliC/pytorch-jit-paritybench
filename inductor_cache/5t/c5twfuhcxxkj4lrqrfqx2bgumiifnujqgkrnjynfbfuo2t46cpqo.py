# AOT ID: ['7_forward']
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


# kernel path: inductor_cache/2g/c2gysfxrf5d77vpntzaiyxcw44zsg4rae25jzggzl65u7tgreig4.py
# Topologically Sorted Source Nodes: [conv2d, group_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   conv2d => convolution
#   group_norm => add, rsqrt, var_mean
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_red_fused_convolution_native_group_norm_0 = async_compile.triton('triton_red_fused_convolution_native_group_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tmp7 = 2048.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cdd5kx3h4nf7ktsyxllzdabsrczhjo3e5uxaxpugxlbji2qidtza.py
# Topologically Sorted Source Nodes: [group_norm, x, out1, input_1], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm => add_1, mul_1
#   input_1 => add_8, rsqrt_4, var_mean_4
#   out1 => add_2, rsqrt_1, var_mean_1
#   x => relu
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
triton_red_fused_native_group_norm_relu_1 = async_compile.triton('triton_red_fused_native_group_norm_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    x0 = (xindex % 32)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 2048.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full([1, 1], 0, tl.int32)
        tmp15 = triton_helpers.maximum(tmp14, tmp13)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r5 + 2048*x4), tmp15, rmask & xmask)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp17, xmask)
    tl.store(out_ptr2 + (x4), tmp18, xmask)
    tl.store(out_ptr3 + (x4), tmp17, xmask)
    tl.store(out_ptr4 + (x4), tmp18, xmask)
    tmp20 = 2048.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(out_ptr5 + (x4), tmp24, xmask)
    tl.store(out_ptr6 + (x4), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kb/ckbbvxqyjulwlktbkvhqggmani3l6wigesrncoj43z3am4thxhkc.py
# Topologically Sorted Source Nodes: [out1, out1_1, input_1, input_2], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_9, mul_9
#   input_2 => relu_4
#   out1 => add_3, mul_3
#   out1_1 => relu_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %unsqueeze_11), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_8), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %unsqueeze_29), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_26), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused_native_group_norm_relu_2 = async_compile.triton('triton_poi_fused_native_group_norm_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x4 // 2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x4 // 2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp17 = tmp0 - tmp16
    tmp19 = tmp18 / tmp4
    tmp20 = tmp19 + tmp6
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp17 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = triton_helpers.maximum(tmp14, tmp26)
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/ut/cutj3rxinh7q7egfza3s2fr3yscirldvck7nsejd6yzkctloumtr.py
# Topologically Sorted Source Nodes: [out2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out2 => add_4, rsqrt_2, var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
triton_red_fused_native_group_norm_3 = async_compile.triton('triton_red_fused_native_group_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtc2ekwdiu6fv55xdba3cz5uf53ad7qq473vsjqqskey2j2forj.py
# Topologically Sorted Source Nodes: [out2, out2_1], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out2 => add_5, mul_5
#   out2_1 => relu_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_17), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_14), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused_native_group_norm_relu_4 = async_compile.triton('triton_poi_fused_native_group_norm_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/u7/cu7vqwnwej45aapzvgdjtxmzg2brfdqvp5u5pl5fiwwxevwaipes.py
# Topologically Sorted Source Nodes: [out3, out3_1], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out3 => add_6, add_7, mul_7, rsqrt_3, var_mean_3
#   out3_1 => relu_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_6, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %unsqueeze_23), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_20), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_per_fused_native_group_norm_relu_5 = async_compile.triton('triton_per_fused_native_group_norm_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp21 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 1024.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr2 + (r1 + 1024*x0), tmp26, None)
    tl.store(out_ptr3 + (x0), tmp19, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/6d/c6dkrtzriiiibq2xvrrjsnp35xafoem3lmnaqdasly4jmsjtnpm7.py
# Topologically Sorted Source Nodes: [out3_3, out3_4], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out3_3 => cat
#   out3_4 => add_10
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_1, %convolution_2, %convolution_3], 1), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %convolution_4), kwargs = {})
triton_poi_fused_add_cat_6 = async_compile.triton('triton_poi_fused_add_cat_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 128)
    x0 = (xindex % 1024)
    x2 = xindex // 131072
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 65536*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-64) + x1) + 32768*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 1024*((-96) + x1) + 32768*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/r3/cr3pylpwulyboeq5xoewxptx33nxuxxmsninhp4hwmv5qvauxh6a.py
# Topologically Sorted Source Nodes: [x_1, out1_3], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_3 => add_11, rsqrt_5, var_mean_5
#   x_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_10, [2, 2], [2, 2]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_10, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
triton_red_fused_avg_pool2d_native_group_norm_7 = async_compile.triton('triton_red_fused_avg_pool2d_native_group_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_native_group_norm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_avg_pool2d_native_group_norm_7(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = (rindex % 16)
        r2 = rindex // 16
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (2*r1 + 64*r2 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*r1 + 64*r2 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (32 + 2*r1 + 64*r2 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (33 + 2*r1 + 64*r2 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 + tmp0
        tmp4 = tmp3 + tmp2
        tmp6 = tmp5 + tmp4
        tmp7 = 0.25
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(out_ptr0 + (r3 + 1024*x0), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tmp13 = 1024.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr3 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ip/ciplnz5p5u6pt72iiopwvph72mec3hlvar4wavsdikevsawnwmna.py
# Topologically Sorted Source Nodes: [out1_3, out1_4], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out1_3 => add_12, mul_11
#   out1_4 => relu_5
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %unsqueeze_35), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_32), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused_native_group_norm_relu_8 = async_compile.triton('triton_poi_fused_native_group_norm_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/o7/co77rdaskg44qv42bl73hpocyes3hp5pj7t3y5qh5dolyokllusw.py
# Topologically Sorted Source Nodes: [out2_3], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out2_3 => add_13, rsqrt_6, var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
triton_per_fused_native_group_norm_9 = async_compile.triton('triton_per_fused_native_group_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_9(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvoaupgrslhovqcwgr4swyn6ozv4oont5d2ck6vtay2ydtlnoigw.py
# Topologically Sorted Source Nodes: [out2_3, out2_4], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out2_3 => add_14, mul_13
#   out2_4 => relu_6
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %unsqueeze_41), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_38), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
triton_poi_fused_native_group_norm_relu_10 = async_compile.triton('triton_poi_fused_native_group_norm_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/o2/co2l7jhoqwuj63tjikgh5ieghairbac4tta5mmrig7bbh6c7hvxc.py
# Topologically Sorted Source Nodes: [out3_5, out3_6], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out3_5 => add_15, add_16, mul_15, rsqrt_7, var_mean_7
#   out3_6 => relu_7
# Graph fragment:
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_14, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %unsqueeze_47), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_44), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_per_fused_native_group_norm_relu_11 = async_compile.triton('triton_per_fused_native_group_norm_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_relu_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp21 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 256.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tl.store(out_ptr2 + (r1 + 256*x0), tmp26, None)
    tl.store(out_ptr3 + (x0), tmp19, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3eav24kuket4i2vyu5676p7x6sryfjgbrq7tthhpiisglrbvc2v.py
# Topologically Sorted Source Nodes: [out3_8, out1_6], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_6 => add_18, rsqrt_8, var_mean_8
#   out3_8 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_5, %convolution_6, %convolution_7], 1), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
triton_per_fused_cat_native_group_norm_12 = async_compile.triton('triton_per_fused_cat_native_group_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_group_norm_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_group_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex // 256
    x0 = (xindex % 32)
    r2 = (rindex % 256)
    x1 = xindex // 32
    r5 = rindex
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (r5 + 1024*x4), None)
    tmp0 = r3 + 4*x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 256*(r3 + 4*x0) + 16384*x1), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (r2 + 256*((-64) + r3 + 4*x0) + 8192*x1), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (r2 + 256*((-96) + r3 + 4*x0) + 8192*x1), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 1024, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 1024.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tl.store(out_ptr0 + (r5 + 1024*x4), tmp16, None)
    tl.store(out_ptr3 + (x4), tmp36, None)
    tl.store(out_ptr1 + (x4), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/o5/co5c3z4qdqk7mmdlusrhjxgumbxh5d2gxhphe2qtfrnz7bil2dn3.py
# Topologically Sorted Source Nodes: [out1_6, out1_7, input_4, input_5], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_25, mul_23
#   input_5 => relu_11
#   out1_6 => add_19, mul_17
#   out1_7 => relu_8
# Graph fragment:
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_53), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_50), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_71), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_68), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
triton_poi_fused_native_group_norm_relu_13 = async_compile.triton('triton_poi_fused_native_group_norm_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp11 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = triton_helpers.maximum(tmp16, tmp21)
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/q2/cq2wdb55etx3iofsloak3bca5ghvqzdji73ypn7omcnhbvgerydo.py
# Topologically Sorted Source Nodes: [out2_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out2_6 => add_20, rsqrt_9, var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
triton_per_fused_native_group_norm_14 = async_compile.triton('triton_per_fused_native_group_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_14(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2sfknu7dsf5fzikukfdkybsa2kxeqklcapjf7k34ke4was26zkh.py
# Topologically Sorted Source Nodes: [out3_13, out3_14, out1_9], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_9 => add_27, rsqrt_12, var_mean_12
#   out3_13 => cat_2
#   out3_14 => add_26
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_8, %convolution_9, %convolution_10], 1), kwargs = {})
#   %add_26 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_2, %convolution_11), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
triton_red_fused_add_cat_native_group_norm_15 = async_compile.triton('triton_red_fused_add_cat_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_group_norm_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_group_norm_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex // 256
        r2 = (rindex % 256)
        r5 = rindex
        tmp17 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r3 + 8*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 128, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r2 + 256*(r3 + 8*x0) + 32768*x1), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 192, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + (r2 + 256*((-128) + r3 + 8*x0) + 16384*x1), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 256, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tl.load(in_ptr2 + (r2 + 256*((-192) + r3 + 8*x0) + 16384*x1), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.where(tmp9, tmp10, tmp14)
        tmp16 = tl.where(tmp4, tmp5, tmp15)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp18, rmask & xmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp20, xmask)
    tl.store(out_ptr1 + (x4), tmp21, xmask)
    tmp23 = 2048.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.store(out_ptr2 + (x4), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sg/csggxjeympohggxjiovpqzzkw5jesizoxo7jzbcm3vbzi3na3jjl.py
# Topologically Sorted Source Nodes: [out1_9, out1_10], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out1_10 => relu_12
#   out1_9 => add_28, mul_25
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %unsqueeze_77), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_74), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused_native_group_norm_relu_16 = async_compile.triton('triton_poi_fused_native_group_norm_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/m3/cm3tzt4o3gabxzg5ywte3rebsnjyrnhzep4xwf2ajb2aqnsqepab.py
# Topologically Sorted Source Nodes: [low1, out1_12], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
# Source node to ATen node mapping:
#   low1 => avg_pool2d_1
#   out1_12 => add_34, rsqrt_15, var_mean_15
# Graph fragment:
#   %avg_pool2d_1 : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_26, [2, 2], [2, 2]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
triton_red_fused_avg_pool2d_native_group_norm_17 = async_compile.triton('triton_red_fused_avg_pool2d_native_group_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_native_group_norm_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_avg_pool2d_native_group_norm_17(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = (rindex % 8)
        r2 = rindex // 8
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (2*r1 + 32*r2 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*r1 + 32*r2 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (16 + 2*r1 + 32*r2 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (17 + 2*r1 + 32*r2 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 + tmp0
        tmp4 = tmp3 + tmp2
        tmp6 = tmp5 + tmp4
        tmp7 = 0.25
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(out_ptr0 + (r3 + 512*x0), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr3 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dz/cdzbparr7fmskkfdcaosmckxkqj3x5b27tyhnrqky77jpk6p4g2n.py
# Topologically Sorted Source Nodes: [out1_12, out1_13], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out1_12 => add_35, mul_31
#   out1_13 => relu_15
# Graph fragment:
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_31, %unsqueeze_95), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_92), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
triton_poi_fused_native_group_norm_relu_18 = async_compile.triton('triton_poi_fused_native_group_norm_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ai/caib2vy3fhyklzsofjx3s3em2gfnahvdh65ecuadkesk6m7yzoob.py
# Topologically Sorted Source Nodes: [out2_12], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out2_12 => add_36, rsqrt_16, var_mean_16
# Graph fragment:
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_16 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
triton_per_fused_native_group_norm_19 = async_compile.triton('triton_per_fused_native_group_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_19(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/mn/cmnjjugp2oxwxtuyvvmvjnihqqmqszmsyintabqyctziahp2knm2.py
# Topologically Sorted Source Nodes: [out2_12, out2_13], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out2_12 => add_37, mul_33
#   out2_13 => relu_16
# Graph fragment:
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %unsqueeze_101), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %unsqueeze_98), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
triton_poi_fused_native_group_norm_relu_20 = async_compile.triton('triton_poi_fused_native_group_norm_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/uv/cuvzyfr2qrymhrmvet5wjktj7vhg5hvhyx5at5qgjgpvhi5nlryi.py
# Topologically Sorted Source Nodes: [out3_20], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out3_20 => add_38, rsqrt_17, var_mean_17
# Graph fragment:
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_34, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-05), kwargs = {})
#   %rsqrt_17 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
triton_per_fused_native_group_norm_21 = async_compile.triton('triton_per_fused_native_group_norm_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_21(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ku/ckuvupzst7hbnosk6p5jq47mrs7kdsfktapbodbyjg4w6dgqz47k.py
# Topologically Sorted Source Nodes: [out3_20, out3_21], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out3_20 => add_39, mul_35
#   out3_21 => relu_17
# Graph fragment:
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, %unsqueeze_107), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_104), kwargs = {})
#   %relu_17 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_39,), kwargs = {})
triton_poi_fused_native_group_norm_relu_22 = async_compile.triton('triton_poi_fused_native_group_norm_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xfkqkqr3zzwcikmcmcldaarop4bubgh6xnh3mnf7i4gjthblie.py
# Topologically Sorted Source Nodes: [out3_23, out3_24, out1_15], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_15 => add_41, rsqrt_18, var_mean_18
#   out3_23 => cat_4
#   out3_24 => add_40
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_15, %convolution_16, %convolution_17], 1), kwargs = {})
#   %add_40 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_4, %avg_pool2d_1), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_36, 1e-05), kwargs = {})
#   %rsqrt_18 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
triton_per_fused_add_cat_native_group_norm_23 = async_compile.triton('triton_per_fused_add_cat_native_group_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_group_norm_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_group_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex // 64
    x0 = (xindex % 32)
    r2 = (rindex % 64)
    x1 = xindex // 32
    r5 = rindex
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (r5 + 512*x4), None)
    tmp0 = r3 + 8*x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 64*(r3 + 8*x0) + 8192*x1), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (r2 + 64*((-128) + r3 + 8*x0) + 4096*x1), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (r2 + 64*((-192) + r3 + 8*x0) + 4096*x1), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 512.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tl.store(out_ptr0 + (r5 + 512*x4), tmp18, None)
    tl.store(out_ptr3 + (x4), tmp36, None)
    tl.store(out_ptr1 + (x4), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wz7ctqcquzljbcks5u73pwutuwwyrsyvfne7nlac53bvasgdos.py
# Topologically Sorted Source Nodes: [low1_1, out1_18], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
# Source node to ATen node mapping:
#   low1_1 => avg_pool2d_2
#   out1_18 => add_48, rsqrt_21, var_mean_21
# Graph fragment:
#   %avg_pool2d_2 : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_40, [2, 2], [2, 2]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_42, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
triton_red_fused_avg_pool2d_native_group_norm_24 = async_compile.triton('triton_red_fused_avg_pool2d_native_group_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_native_group_norm_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_avg_pool2d_native_group_norm_24(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = (rindex % 4)
        r2 = rindex // 4
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (2*r1 + 16*r2 + 512*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*r1 + 16*r2 + 512*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (8 + 2*r1 + 16*r2 + 512*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (9 + 2*r1 + 16*r2 + 512*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 + tmp0
        tmp4 = tmp3 + tmp2
        tmp6 = tmp5 + tmp4
        tmp7 = 0.25
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(out_ptr0 + (r3 + 128*x0), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tmp13 = 128.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr3 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gk/cgks2v7infjmb4rgtv23blklhjx2xvg6ouv4etle5qjo3lpidhur.py
# Topologically Sorted Source Nodes: [out1_18, out1_19], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out1_18 => add_49, mul_43
#   out1_19 => relu_21
# Graph fragment:
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_43, %unsqueeze_131), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_128), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_49,), kwargs = {})
triton_poi_fused_native_group_norm_relu_25 = async_compile.triton('triton_poi_fused_native_group_norm_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ua/cua5tckppr3h6466nwz5l7sze2hvjcpjcz5tkkazkoha75qf4ndh.py
# Topologically Sorted Source Nodes: [out2_18], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out2_18 => add_50, rsqrt_22, var_mean_22
# Graph fragment:
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_44, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
triton_per_fused_native_group_norm_26 = async_compile.triton('triton_per_fused_native_group_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_26(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/au/cau46yoqizkwdalr2od5pd3lapfs6dsvcrjj726uhxq23djwhk65.py
# Topologically Sorted Source Nodes: [out2_18, out2_19], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out2_18 => add_51, mul_45
#   out2_19 => relu_22
# Graph fragment:
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %unsqueeze_137), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_134), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_51,), kwargs = {})
triton_poi_fused_native_group_norm_relu_27 = async_compile.triton('triton_poi_fused_native_group_norm_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 64.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/vr/cvrzc52y3f2llkmt2yfomjg3e77by66lifg6g4gs5g5btxeqjlbi.py
# Topologically Sorted Source Nodes: [out3_30], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   out3_30 => add_52, rsqrt_23, var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_46, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_23 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_52,), kwargs = {})
triton_per_fused_native_group_norm_28 = async_compile.triton('triton_per_fused_native_group_norm_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_28(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 32, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 32.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/in/cinwnytpjfj5z4b5pknvwcakyskfcewpq7r36xujoat54vv3yrq4.py
# Topologically Sorted Source Nodes: [out3_30, out3_31], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out3_30 => add_53, mul_47
#   out3_31 => relu_23
# Graph fragment:
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_47, %unsqueeze_143), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_140), kwargs = {})
#   %relu_23 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused_native_group_norm_relu_29 = async_compile.triton('triton_poi_fused_native_group_norm_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/3y/c3y3yz5sfk7oz6kcvh2rrfcgq3jb3b42pko7ublpe6cptugbwbcj.py
# Topologically Sorted Source Nodes: [out3_33, out1_21], Original ATen: [aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_21 => add_55, rsqrt_24, var_mean_24
#   out3_33 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_21, %convolution_22, %convolution_23], 1), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
triton_per_fused_cat_native_group_norm_30 = async_compile.triton('triton_per_fused_cat_native_group_norm_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_group_norm_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_group_norm_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex // 16
    x0 = (xindex % 32)
    r2 = (rindex % 16)
    x1 = xindex // 32
    r5 = rindex
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (r5 + 128*x4), xmask, other=0.0)
    tmp0 = r3 + 8*x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 16*(r3 + 8*x0) + 2048*x1), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (r2 + 16*((-128) + r3 + 8*x0) + 1024*x1), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (r2 + 16*((-192) + r3 + 8*x0) + 1024*x1), tmp11 & xmask, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = 128.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tl.store(out_ptr0 + (r5 + 128*x4), tmp16, xmask)
    tl.store(out_ptr3 + (x4), tmp39, xmask)
    tl.store(out_ptr1 + (x4), tmp28, xmask)
    tl.store(out_ptr2 + (x4), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cl/cclcf37rjgiys6wpmkjdandn4bz4rj3gudtgynvxl2amqnqdy4gv.py
# Topologically Sorted Source Nodes: [out1_21, out1_22], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   out1_21 => add_56, mul_49
#   out1_22 => relu_24
# Graph fragment:
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_49, %unsqueeze_149), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_146), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
triton_poi_fused_native_group_norm_relu_31 = async_compile.triton('triton_poi_fused_native_group_norm_relu_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 8), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/vr/cvrjszbwmee3ndhwzicg77dkpdoe57l4hxi6qiya6hgzo4czsjbe.py
# Topologically Sorted Source Nodes: [out3_34, out3_38, out3_39, out1_24], Original ATen: [aten.add, aten.cat, aten.native_group_norm]
# Source node to ATen node mapping:
#   out1_24 => add_62, rsqrt_27, var_mean_27
#   out3_34 => add_54
#   out3_38 => cat_7
#   out3_39 => add_61
# Graph fragment:
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_6, %avg_pool2d_2), kwargs = {})
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_24, %convolution_25, %convolution_26], 1), kwargs = {})
#   %add_61 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_7, %add_54), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_54, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_62,), kwargs = {})
triton_per_fused_add_cat_native_group_norm_32 = async_compile.triton('triton_per_fused_add_cat_native_group_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_group_norm_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex // 16
    x0 = (xindex % 32)
    r2 = (rindex % 16)
    x1 = xindex // 32
    r5 = rindex
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (r5 + 128*x4), xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r5 + 128*x4), xmask, other=0.0)
    tmp0 = r3 + 8*x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 16*(r3 + 8*x0) + 2048*x1), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (r2 + 16*((-128) + r3 + 8*x0) + 1024*x1), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (r2 + 16*((-192) + r3 + 8*x0) + 1024*x1), tmp11 & xmask, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = 128.0
    tmp38 = tmp36 / tmp37
    tmp39 = 1e-05
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tl.store(out_ptr0 + (r5 + 128*x4), tmp20, xmask)
    tl.store(out_ptr3 + (x4), tmp41, xmask)
    tl.store(out_ptr1 + (x4), tmp30, xmask)
    tl.store(out_ptr2 + (x4), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ut/cutlgkmmn6fln4znqbucluuj3miovgjdtj2dw5g6mcjcoy57tj2q.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up2 => clamp_max_2, clamp_min_2, convert_element_type_3, floor_1, sub_32
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_180,), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_32, 0), kwargs = {})
#   %clamp_max_2 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 3), kwargs = {})
triton_poi_fused__to_copy_clamp_floor_sub_33 = async_compile.triton('triton_poi_fused__to_copy_clamp_floor_sub_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_floor_sub_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_floor_sub_33(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citjburegcbiykksbqlnm37wd7xz7mukubtydszdi5xrwnlaqv7c.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   up2 => clamp_max_5, clamp_min_5, convert_element_type, convert_element_type_2, floor, iota, mul_60
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_60 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.42857142857142855), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_60,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 0), kwargs = {})
#   %clamp_max_5 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 3), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_floor_mul_34 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_floor_mul_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_floor_mul_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_floor_mul_34(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1], 3, tl.int64)
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3s/c3s3hr6cgvnio6szhdc2hpdujln5nv73euoh4iwxzz2l64uv2l43.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2 => add_71, clamp_max_7, clamp_min_7, convert_element_type, convert_element_type_2, floor, iota, mul_60
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_60 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.42857142857142855), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_60,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_71, 0), kwargs = {})
#   %clamp_max_7 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 3), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdidoivsqp2tp2xuhnqmmc2ndrjff5ywrnm3njzu5gots6t6tgtu.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2 => add_72, clamp_max_9, clamp_min_9, convert_element_type, convert_element_type_2, floor, iota, mul_60
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_60 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.42857142857142855), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_60,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 2), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_72, 0), kwargs = {})
#   %clamp_max_9 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 3), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w2/cw2wy7j5az27icqpcw4kter4ilhne3xydkc4siocpk4cfid2x3ub.py
# Topologically Sorted Source Nodes: [out3_28, out3_29, out3_43, out3_44, up2, low2, out1_27], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   low2 => add_98
#   out1_27 => add_99, rsqrt_30, var_mean_30
#   out3_28 => cat_5
#   out3_29 => add_47
#   out3_43 => cat_8
#   out3_44 => add_68
#   up2 => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_73, add_74, add_75, add_76, add_77, add_78, add_79, add_80, add_81, add_82, add_83, add_84, add_85, add_86, add_87, add_88, add_89, add_90, add_91, add_92, add_93, add_94, add_95, add_96, add_97, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type, floor, floor_1, iota, mul_100, mul_101, mul_102, mul_103, mul_104, mul_105, mul_60, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, mul_88, mul_89, mul_90, mul_91, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, mul_98, mul_99, sub_30, sub_31, sub_34, sub_35, sub_36, sub_37, sub_38, sub_39, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45, sub_46, sub_47, sub_48, sub_49
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_18, %convolution_19, %convolution_20], 1), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_5, %add_40), kwargs = {})
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_27, %convolution_28, %convolution_29], 1), kwargs = {})
#   %add_68 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_8, %add_61), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_60 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.42857142857142855), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_60,), kwargs = {})
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_180,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_180, %floor_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_30, 0.0), kwargs = {})
#   %clamp_max : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_60, %floor), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_31, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1.0), kwargs = {})
#   %add_73 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, 1.0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, -0.75), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_62, -3.75), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %add_73), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, -6.0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %add_73), kwargs = {})
#   %sub_35 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_64, -3.0), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 1.25), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_65, 2.25), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_1), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %clamp_max_1), kwargs = {})
#   %add_75 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, 1), kwargs = {})
#   %sub_37 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_1), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, 1.25), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_68, 2.25), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %sub_37), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %sub_37), kwargs = {})
#   %add_76 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, 1), kwargs = {})
#   %sub_39 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_1), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, -0.75), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_71, -3.75), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %sub_39), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, -6.0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, %sub_39), kwargs = {})
#   %sub_41 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_73, -3.0), kwargs = {})
#   %add_78 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, 1.0), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, -0.75), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_74, -3.75), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %add_78), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, -6.0), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %add_78), kwargs = {})
#   %sub_43 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_76, -3.0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 1.25), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_77, 2.25), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %clamp_max), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %clamp_max), kwargs = {})
#   %add_80 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, 1), kwargs = {})
#   %sub_45 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, 1.25), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_80, 2.25), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %sub_45), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %sub_45), kwargs = {})
#   %add_81 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, 1), kwargs = {})
#   %sub_47 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, -0.75), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, -3.75), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %sub_47), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, -6.0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %sub_47), kwargs = {})
#   %sub_49 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_85, -3.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_2, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_2, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_2, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_2, %clamp_max_9]), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, %sub_35), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_1, %add_75), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %mul_87), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %add_76), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %mul_88), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_3, %sub_41), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %mul_89), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_10, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_10, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_10, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_10, %clamp_max_9]), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_4, %sub_35), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_5, %add_75), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_91), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_6, %add_76), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %mul_92), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_7, %sub_41), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %mul_93), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_18, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_18, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_18, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_18, %clamp_max_9]), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_8, %sub_35), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_9, %add_75), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %mul_95), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_10, %add_76), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %mul_96), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_11, %sub_41), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %mul_97), kwargs = {})
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_26, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_26, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_26, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_68, [None, None, %clamp_max_26, %clamp_max_9]), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_12, %sub_35), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_13, %add_75), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %mul_99), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_14, %add_76), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %mul_100), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_15, %sub_41), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %mul_101), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %sub_43), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_88, %add_80), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, %mul_103), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %add_81), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %mul_104), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %sub_49), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %mul_105), kwargs = {})
#   %add_98 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %add_97), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_60, 1e-05), kwargs = {})
#   %rsqrt_30 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_99,), kwargs = {})
triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_37 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr15, out_ptr16, out_ptr17, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x5 = xindex
    tmp275_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = ((rindex // 8) % 8)
        r2 = (rindex % 8)
        r4 = rindex // 64
        r7 = rindex
        r6 = (rindex % 64)
        tmp0 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp87 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp107 = tl.load(in_ptr9 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp143 = tl.load(in_ptr10 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp179 = tl.load(in_ptr11 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp271 = tl.load(in_ptr15 + (r7 + 512*x5), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp9 = r4 + 8*x0
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tl.full([1, 1], 128, tl.int64)
        tmp13 = tmp9 < tmp12
        tmp14 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp9 >= tmp12
        tmp16 = tl.full([1, 1], 192, tl.int64)
        tmp17 = tmp9 < tmp16
        tmp18 = tmp15 & tmp17
        tmp19 = tl.load(in_ptr3 + (tmp8 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp9 >= tmp16
        tmp21 = tl.full([1, 1], 256, tl.int64)
        tmp22 = tmp9 < tmp21
        tmp23 = tl.load(in_ptr4 + (tmp8 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.where(tmp18, tmp19, tmp23)
        tmp25 = tl.where(tmp13, tmp14, tmp24)
        tmp26 = tl.load(in_ptr5 + (tmp8 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 + tmp26
        tmp28 = r2
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 0.42857142857142855
        tmp31 = tmp29 * tmp30
        tmp32 = libdevice.floor(tmp31)
        tmp33 = tmp31 - tmp32
        tmp34 = 0.0
        tmp35 = triton_helpers.maximum(tmp33, tmp34)
        tmp36 = 1.0
        tmp37 = triton_helpers.minimum(tmp35, tmp36)
        tmp38 = tmp37 + tmp36
        tmp39 = -0.75
        tmp40 = tmp38 * tmp39
        tmp41 = -3.75
        tmp42 = tmp40 - tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = -6.0
        tmp45 = tmp43 + tmp44
        tmp46 = tmp45 * tmp38
        tmp47 = -3.0
        tmp48 = tmp46 - tmp47
        tmp49 = tmp27 * tmp48
        tmp51 = tmp50 + tmp1
        tmp52 = tmp50 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp50)
        tmp54 = tl.load(in_ptr2 + (tmp53 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr3 + (tmp53 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr4 + (tmp53 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.where(tmp18, tmp55, tmp56)
        tmp58 = tl.where(tmp13, tmp54, tmp57)
        tmp59 = tl.load(in_ptr5 + (tmp53 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp58 + tmp59
        tmp61 = 1.25
        tmp62 = tmp37 * tmp61
        tmp63 = 2.25
        tmp64 = tmp62 - tmp63
        tmp65 = tmp64 * tmp37
        tmp66 = tmp65 * tmp37
        tmp67 = tmp66 + tmp36
        tmp68 = tmp60 * tmp67
        tmp70 = tmp69 + tmp1
        tmp71 = tmp69 < 0
        tmp72 = tl.where(tmp71, tmp70, tmp69)
        tmp73 = tl.load(in_ptr2 + (tmp72 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp74 = tl.load(in_ptr3 + (tmp72 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp75 = tl.load(in_ptr4 + (tmp72 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.where(tmp18, tmp74, tmp75)
        tmp77 = tl.where(tmp13, tmp73, tmp76)
        tmp78 = tl.load(in_ptr5 + (tmp72 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tmp77 + tmp78
        tmp80 = tmp36 - tmp37
        tmp81 = tmp80 * tmp61
        tmp82 = tmp81 - tmp63
        tmp83 = tmp82 * tmp80
        tmp84 = tmp83 * tmp80
        tmp85 = tmp84 + tmp36
        tmp86 = tmp79 * tmp85
        tmp88 = tmp87 + tmp1
        tmp89 = tmp87 < 0
        tmp90 = tl.where(tmp89, tmp88, tmp87)
        tmp91 = tl.load(in_ptr2 + (tmp90 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.load(in_ptr3 + (tmp90 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = tl.load(in_ptr4 + (tmp90 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tl.where(tmp18, tmp92, tmp93)
        tmp95 = tl.where(tmp13, tmp91, tmp94)
        tmp96 = tl.load(in_ptr5 + (tmp90 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp97 = tmp95 + tmp96
        tmp98 = 2.0
        tmp99 = tmp98 - tmp37
        tmp100 = tmp99 * tmp39
        tmp101 = tmp100 - tmp41
        tmp102 = tmp101 * tmp99
        tmp103 = tmp102 + tmp44
        tmp104 = tmp103 * tmp99
        tmp105 = tmp104 - tmp47
        tmp106 = tmp97 * tmp105
        tmp108 = tmp107 + tmp1
        tmp109 = tmp107 < 0
        tmp110 = tl.where(tmp109, tmp108, tmp107)
        tmp111 = tl.load(in_ptr2 + (tmp8 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp112 = tl.load(in_ptr3 + (tmp8 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp113 = tl.load(in_ptr4 + (tmp8 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp114 = tl.where(tmp18, tmp112, tmp113)
        tmp115 = tl.where(tmp13, tmp111, tmp114)
        tmp116 = tl.load(in_ptr5 + (tmp8 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp117 = tmp115 + tmp116
        tmp118 = tmp117 * tmp48
        tmp119 = tl.load(in_ptr2 + (tmp53 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp120 = tl.load(in_ptr3 + (tmp53 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp121 = tl.load(in_ptr4 + (tmp53 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp122 = tl.where(tmp18, tmp120, tmp121)
        tmp123 = tl.where(tmp13, tmp119, tmp122)
        tmp124 = tl.load(in_ptr5 + (tmp53 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tmp123 + tmp124
        tmp126 = tmp125 * tmp67
        tmp127 = tl.load(in_ptr2 + (tmp72 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp128 = tl.load(in_ptr3 + (tmp72 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.load(in_ptr4 + (tmp72 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp130 = tl.where(tmp18, tmp128, tmp129)
        tmp131 = tl.where(tmp13, tmp127, tmp130)
        tmp132 = tl.load(in_ptr5 + (tmp72 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp133 = tmp131 + tmp132
        tmp134 = tmp133 * tmp85
        tmp135 = tl.load(in_ptr2 + (tmp90 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp136 = tl.load(in_ptr3 + (tmp90 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp137 = tl.load(in_ptr4 + (tmp90 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp138 = tl.where(tmp18, tmp136, tmp137)
        tmp139 = tl.where(tmp13, tmp135, tmp138)
        tmp140 = tl.load(in_ptr5 + (tmp90 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp141 = tmp139 + tmp140
        tmp142 = tmp141 * tmp105
        tmp144 = tmp143 + tmp1
        tmp145 = tmp143 < 0
        tmp146 = tl.where(tmp145, tmp144, tmp143)
        tmp147 = tl.load(in_ptr2 + (tmp8 + 4*tmp146 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp148 = tl.load(in_ptr3 + (tmp8 + 4*tmp146 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp149 = tl.load(in_ptr4 + (tmp8 + 4*tmp146 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.where(tmp18, tmp148, tmp149)
        tmp151 = tl.where(tmp13, tmp147, tmp150)
        tmp152 = tl.load(in_ptr5 + (tmp8 + 4*tmp146 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp153 = tmp151 + tmp152
        tmp154 = tmp153 * tmp48
        tmp155 = tl.load(in_ptr2 + (tmp53 + 4*tmp146 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp156 = tl.load(in_ptr3 + (tmp53 + 4*tmp146 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp157 = tl.load(in_ptr4 + (tmp53 + 4*tmp146 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp158 = tl.where(tmp18, tmp156, tmp157)
        tmp159 = tl.where(tmp13, tmp155, tmp158)
        tmp160 = tl.load(in_ptr5 + (tmp53 + 4*tmp146 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp161 = tmp159 + tmp160
        tmp162 = tmp161 * tmp67
        tmp163 = tl.load(in_ptr2 + (tmp72 + 4*tmp146 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp164 = tl.load(in_ptr3 + (tmp72 + 4*tmp146 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp165 = tl.load(in_ptr4 + (tmp72 + 4*tmp146 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp166 = tl.where(tmp18, tmp164, tmp165)
        tmp167 = tl.where(tmp13, tmp163, tmp166)
        tmp168 = tl.load(in_ptr5 + (tmp72 + 4*tmp146 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp169 * tmp85
        tmp171 = tl.load(in_ptr2 + (tmp90 + 4*tmp146 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp172 = tl.load(in_ptr3 + (tmp90 + 4*tmp146 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp173 = tl.load(in_ptr4 + (tmp90 + 4*tmp146 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp174 = tl.where(tmp18, tmp172, tmp173)
        tmp175 = tl.where(tmp13, tmp171, tmp174)
        tmp176 = tl.load(in_ptr5 + (tmp90 + 4*tmp146 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp177 = tmp175 + tmp176
        tmp178 = tmp177 * tmp105
        tmp180 = tmp179 + tmp1
        tmp181 = tmp179 < 0
        tmp182 = tl.where(tmp181, tmp180, tmp179)
        tmp183 = tl.load(in_ptr2 + (tmp8 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp184 = tl.load(in_ptr3 + (tmp8 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp185 = tl.load(in_ptr4 + (tmp8 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp186 = tl.where(tmp18, tmp184, tmp185)
        tmp187 = tl.where(tmp13, tmp183, tmp186)
        tmp188 = tl.load(in_ptr5 + (tmp8 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp189 = tmp187 + tmp188
        tmp190 = tmp189 * tmp48
        tmp191 = tl.load(in_ptr2 + (tmp53 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp192 = tl.load(in_ptr3 + (tmp53 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp193 = tl.load(in_ptr4 + (tmp53 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp194 = tl.where(tmp18, tmp192, tmp193)
        tmp195 = tl.where(tmp13, tmp191, tmp194)
        tmp196 = tl.load(in_ptr5 + (tmp53 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp197 = tmp195 + tmp196
        tmp198 = tmp197 * tmp67
        tmp199 = tl.load(in_ptr2 + (tmp72 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp200 = tl.load(in_ptr3 + (tmp72 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp201 = tl.load(in_ptr4 + (tmp72 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp202 = tl.where(tmp18, tmp200, tmp201)
        tmp203 = tl.where(tmp13, tmp199, tmp202)
        tmp204 = tl.load(in_ptr5 + (tmp72 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp205 = tmp203 + tmp204
        tmp206 = tmp205 * tmp85
        tmp207 = tl.load(in_ptr2 + (tmp90 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp208 = tl.load(in_ptr3 + (tmp90 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp209 = tl.load(in_ptr4 + (tmp90 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp210 = tl.where(tmp18, tmp208, tmp209)
        tmp211 = tl.where(tmp13, tmp207, tmp210)
        tmp212 = tl.load(in_ptr5 + (tmp90 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp213 = tmp211 + tmp212
        tmp214 = tmp213 * tmp105
        tmp215 = tmp49 + tmp68
        tmp216 = tmp215 + tmp86
        tmp217 = tmp216 + tmp106
        tmp218 = r3
        tmp219 = tmp218.to(tl.float32)
        tmp220 = tmp219 * tmp30
        tmp221 = libdevice.floor(tmp220)
        tmp222 = tmp220 - tmp221
        tmp223 = triton_helpers.maximum(tmp222, tmp34)
        tmp224 = triton_helpers.minimum(tmp223, tmp36)
        tmp225 = tmp224 + tmp36
        tmp226 = tmp225 * tmp39
        tmp227 = tmp226 - tmp41
        tmp228 = tmp227 * tmp225
        tmp229 = tmp228 + tmp44
        tmp230 = tmp229 * tmp225
        tmp231 = tmp230 - tmp47
        tmp232 = tmp217 * tmp231
        tmp233 = tmp118 + tmp126
        tmp234 = tmp233 + tmp134
        tmp235 = tmp234 + tmp142
        tmp236 = tmp224 * tmp61
        tmp237 = tmp236 - tmp63
        tmp238 = tmp237 * tmp224
        tmp239 = tmp238 * tmp224
        tmp240 = tmp239 + tmp36
        tmp241 = tmp235 * tmp240
        tmp242 = tmp232 + tmp241
        tmp243 = tmp154 + tmp162
        tmp244 = tmp243 + tmp170
        tmp245 = tmp244 + tmp178
        tmp246 = tmp36 - tmp224
        tmp247 = tmp246 * tmp61
        tmp248 = tmp247 - tmp63
        tmp249 = tmp248 * tmp246
        tmp250 = tmp249 * tmp246
        tmp251 = tmp250 + tmp36
        tmp252 = tmp245 * tmp251
        tmp253 = tmp242 + tmp252
        tmp254 = tmp190 + tmp198
        tmp255 = tmp254 + tmp206
        tmp256 = tmp255 + tmp214
        tmp257 = tmp98 - tmp224
        tmp258 = tmp257 * tmp39
        tmp259 = tmp258 - tmp41
        tmp260 = tmp259 * tmp257
        tmp261 = tmp260 + tmp44
        tmp262 = tmp261 * tmp257
        tmp263 = tmp262 - tmp47
        tmp264 = tmp256 * tmp263
        tmp265 = tmp253 + tmp264
        tmp266 = tl.load(in_ptr12 + (r6 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp267 = tl.load(in_ptr13 + (r6 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_first', other=0.0)
        tmp268 = tl.load(in_ptr14 + (r6 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_first', other=0.0)
        tmp269 = tl.where(tmp18, tmp267, tmp268)
        tmp270 = tl.where(tmp13, tmp266, tmp269)
        tmp272 = tmp270 + tmp271
        tmp273 = tmp272 + tmp265
        tmp274 = tl.broadcast_to(tmp273, [XBLOCK, RBLOCK])
        tmp275_mean_next, tmp275_m2_next, tmp275_weight_next = triton_helpers.welford_reduce(
            tmp274, tmp275_mean, tmp275_m2, tmp275_weight, roffset == 0
        )
        tmp275_mean = tl.where(rmask & xmask, tmp275_mean_next, tmp275_mean)
        tmp275_m2 = tl.where(rmask & xmask, tmp275_m2_next, tmp275_m2)
        tmp275_weight = tl.where(rmask & xmask, tmp275_weight_next, tmp275_weight)
        tl.store(in_out_ptr0 + (r7 + 512*x5), tmp273, rmask & xmask)
    tmp275_tmp, tmp276_tmp, tmp277_tmp = triton_helpers.welford(
        tmp275_mean, tmp275_m2, tmp275_weight, 1
    )
    tmp275 = tmp275_tmp[:, None]
    tmp276 = tmp276_tmp[:, None]
    tmp277 = tmp277_tmp[:, None]
    tl.store(out_ptr15 + (x5), tmp275, xmask)
    tl.store(out_ptr16 + (x5), tmp276, xmask)
    tmp278 = 512.0
    tmp279 = tmp276 / tmp278
    tmp280 = 1e-05
    tmp281 = tmp279 + tmp280
    tmp282 = libdevice.rsqrt(tmp281)
    tl.store(out_ptr17 + (x5), tmp282, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hp/chpy7hi62lc3bkwmhg7v22wnkwknwiuelc5o6ffpbtmorh447q7s.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => clamp_max_36, clamp_min_36, convert_element_type_7, floor_3, sub_55
# Graph fragment:
#   %floor_3 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_199,), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_3, torch.int64), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, 1), kwargs = {})
#   %clamp_min_36 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_55, 0), kwargs = {})
#   %clamp_max_36 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_36, 7), kwargs = {})
triton_poi_fused__to_copy_clamp_floor_sub_38 = async_compile.triton('triton_poi_fused__to_copy_clamp_floor_sub_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_floor_sub_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_floor_sub_38(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 7, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tz/ctzxnfbn2wad2v7kkyyo5xw6ujol332owil3j7vrhxmyqacidfe5.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => clamp_max_39, clamp_min_39, convert_element_type_4, convert_element_type_6, floor_2, iota_2, mul_112
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.4666666666666667), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_112,), kwargs = {})
#   %convert_element_type_6 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_2, torch.int64), kwargs = {})
#   %clamp_min_39 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_6, 0), kwargs = {})
#   %clamp_max_39 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_39, 7), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_floor_mul_39 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_floor_mul_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_floor_mul_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_floor_mul_39(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1], 7, tl.int64)
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/od/codeclzx7vvc4gzedeneilgks4oztedwei655qktxm3twzzdluey.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => add_108, clamp_max_41, clamp_min_41, convert_element_type_4, convert_element_type_6, floor_2, iota_2, mul_112
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.4666666666666667), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_112,), kwargs = {})
#   %convert_element_type_6 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_2, torch.int64), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 1), kwargs = {})
#   %clamp_min_41 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_108, 0), kwargs = {})
#   %clamp_max_41 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_41, 7), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 7, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/he/cherlp2ohvv4s4tku4yvjkkg7rycsrwmooqgrrd6q3g52rf345yn.py
# Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2_1 => add_109, clamp_max_43, clamp_min_43, convert_element_type_4, convert_element_type_6, floor_2, iota_2, mul_112
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.4666666666666667), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_112,), kwargs = {})
#   %convert_element_type_6 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_2, torch.int64), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 2), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_109, 0), kwargs = {})
#   %clamp_max_43 : [num_users=17] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 7), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 7, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kk/ckkqz3wsk2t3uigjwxye6qzvueu4y6ot3dy2jw6i4fczzrtrzzdq.py
# Topologically Sorted Source Nodes: [out3_18, out3_19, out3_48, out3_49, up2_1, hg, out1_30], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   hg => add_135
#   out1_30 => add_136, rsqrt_33, var_mean_33
#   out3_18 => cat_3
#   out3_19 => add_33
#   out3_48 => cat_9
#   out3_49 => add_105
#   up2_1 => _unsafe_index_16, _unsafe_index_17, _unsafe_index_18, _unsafe_index_19, _unsafe_index_20, _unsafe_index_21, _unsafe_index_22, _unsafe_index_23, _unsafe_index_24, _unsafe_index_25, _unsafe_index_26, _unsafe_index_27, _unsafe_index_28, _unsafe_index_29, _unsafe_index_30, _unsafe_index_31, add_110, add_111, add_112, add_113, add_114, add_115, add_116, add_117, add_118, add_119, add_120, add_121, add_122, add_123, add_124, add_125, add_126, add_127, add_128, add_129, add_130, add_131, add_132, add_133, add_134, clamp_max_34, clamp_max_35, clamp_min_34, clamp_min_35, convert_element_type_4, floor_2, floor_3, iota_2, mul_112, mul_114, mul_115, mul_116, mul_117, mul_118, mul_119, mul_120, mul_121, mul_122, mul_123, mul_124, mul_125, mul_126, mul_127, mul_128, mul_129, mul_130, mul_131, mul_132, mul_133, mul_134, mul_135, mul_136, mul_137, mul_138, mul_139, mul_140, mul_141, mul_142, mul_143, mul_144, mul_145, mul_146, mul_147, mul_148, mul_149, mul_150, mul_151, mul_152, mul_153, mul_154, mul_155, mul_156, mul_157, sub_53, sub_54, sub_57, sub_58, sub_59, sub_60, sub_61, sub_62, sub_63, sub_64, sub_65, sub_66, sub_67, sub_68, sub_69, sub_70, sub_71, sub_72
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_12, %convolution_13, %convolution_14], 1), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_3, %add_26), kwargs = {})
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_30, %convolution_31, %convolution_32], 1), kwargs = {})
#   %add_105 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_9, %add_98), kwargs = {})
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.4666666666666667), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_112,), kwargs = {})
#   %floor_3 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_199,), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_199, %floor_3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_53, 0.0), kwargs = {})
#   %clamp_max_34 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 1.0), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_112, %floor_2), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_54, 0.0), kwargs = {})
#   %clamp_max_35 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 1.0), kwargs = {})
#   %add_110 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_35, 1.0), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, -0.75), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_114, -3.75), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %add_110), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, -6.0), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_111, %add_110), kwargs = {})
#   %sub_58 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_116, -3.0), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_35, 1.25), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_117, 2.25), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %clamp_max_35), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %clamp_max_35), kwargs = {})
#   %add_112 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, 1), kwargs = {})
#   %sub_60 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_35), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, 1.25), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_120, 2.25), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %sub_60), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %sub_60), kwargs = {})
#   %add_113 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, 1), kwargs = {})
#   %sub_62 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_35), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, -0.75), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_123, -3.75), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %sub_62), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, -6.0), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %sub_62), kwargs = {})
#   %sub_64 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_125, -3.0), kwargs = {})
#   %add_115 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_34, 1.0), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, -0.75), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_126, -3.75), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %add_115), kwargs = {})
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, -6.0), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %add_115), kwargs = {})
#   %sub_66 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_128, -3.0), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_34, 1.25), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_129, 2.25), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %clamp_max_34), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %clamp_max_34), kwargs = {})
#   %add_117 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, 1), kwargs = {})
#   %sub_68 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_34), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, 1.25), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_132, 2.25), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %sub_68), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %sub_68), kwargs = {})
#   %add_118 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, 1), kwargs = {})
#   %sub_70 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_34), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, -0.75), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_135, -3.75), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %sub_70), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, -6.0), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %sub_70), kwargs = {})
#   %sub_72 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_137, -3.0), kwargs = {})
#   %_unsafe_index_16 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_36, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_17 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_36, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_18 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_36, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_19 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_36, %clamp_max_43]), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_16, %sub_58), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_17, %add_112), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_138, %mul_139), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_18, %add_113), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_120, %mul_140), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_19, %sub_64), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %mul_141), kwargs = {})
#   %_unsafe_index_20 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_44, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_21 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_44, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_22 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_44, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_23 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_44, %clamp_max_43]), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_20, %sub_58), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_21, %add_112), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_142, %mul_143), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_22, %add_113), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %mul_144), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_23, %sub_64), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_124, %mul_145), kwargs = {})
#   %_unsafe_index_24 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_52, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_25 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_52, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_26 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_52, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_52, %clamp_max_43]), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_24, %sub_58), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_25, %add_112), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %mul_147), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_26, %add_113), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %mul_148), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_27, %sub_64), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %mul_149), kwargs = {})
#   %_unsafe_index_28 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_60, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_29 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_60, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_30 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_60, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_31 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_105, [None, None, %clamp_max_60, %clamp_max_43]), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_28, %sub_58), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_29, %add_112), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %mul_151), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_30, %add_113), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_129, %mul_152), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_31, %sub_64), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_130, %mul_153), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sub_66), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_125, %add_117), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %mul_155), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %add_118), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, %mul_156), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sub_72), kwargs = {})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %mul_157), kwargs = {})
#   %add_135 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %add_134), kwargs = {})
#   %var_mean_33 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_66, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_66, 1e-05), kwargs = {})
#   %rsqrt_33 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_136,), kwargs = {})
triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_42 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr15, out_ptr16, out_ptr17, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x5 = xindex
    tmp275_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = ((rindex // 16) % 16)
        r2 = (rindex % 16)
        r4 = rindex // 256
        r7 = rindex
        r6 = (rindex % 256)
        tmp0 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp87 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp107 = tl.load(in_ptr9 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp143 = tl.load(in_ptr10 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp179 = tl.load(in_ptr11 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp271 = tl.load(in_ptr15 + (r7 + 2048*x5), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp9 = r4 + 8*x0
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tl.full([1, 1], 128, tl.int64)
        tmp13 = tmp9 < tmp12
        tmp14 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp9 >= tmp12
        tmp16 = tl.full([1, 1], 192, tl.int64)
        tmp17 = tmp9 < tmp16
        tmp18 = tmp15 & tmp17
        tmp19 = tl.load(in_ptr3 + (tmp8 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp9 >= tmp16
        tmp21 = tl.full([1, 1], 256, tl.int64)
        tmp22 = tmp9 < tmp21
        tmp23 = tl.load(in_ptr4 + (tmp8 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.where(tmp18, tmp19, tmp23)
        tmp25 = tl.where(tmp13, tmp14, tmp24)
        tmp26 = tl.load(in_ptr5 + (tmp8 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 + tmp26
        tmp28 = r2
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 0.4666666666666667
        tmp31 = tmp29 * tmp30
        tmp32 = libdevice.floor(tmp31)
        tmp33 = tmp31 - tmp32
        tmp34 = 0.0
        tmp35 = triton_helpers.maximum(tmp33, tmp34)
        tmp36 = 1.0
        tmp37 = triton_helpers.minimum(tmp35, tmp36)
        tmp38 = tmp37 + tmp36
        tmp39 = -0.75
        tmp40 = tmp38 * tmp39
        tmp41 = -3.75
        tmp42 = tmp40 - tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = -6.0
        tmp45 = tmp43 + tmp44
        tmp46 = tmp45 * tmp38
        tmp47 = -3.0
        tmp48 = tmp46 - tmp47
        tmp49 = tmp27 * tmp48
        tmp51 = tmp50 + tmp1
        tmp52 = tmp50 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp50)
        tmp54 = tl.load(in_ptr2 + (tmp53 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr3 + (tmp53 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr4 + (tmp53 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.where(tmp18, tmp55, tmp56)
        tmp58 = tl.where(tmp13, tmp54, tmp57)
        tmp59 = tl.load(in_ptr5 + (tmp53 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp58 + tmp59
        tmp61 = 1.25
        tmp62 = tmp37 * tmp61
        tmp63 = 2.25
        tmp64 = tmp62 - tmp63
        tmp65 = tmp64 * tmp37
        tmp66 = tmp65 * tmp37
        tmp67 = tmp66 + tmp36
        tmp68 = tmp60 * tmp67
        tmp70 = tmp69 + tmp1
        tmp71 = tmp69 < 0
        tmp72 = tl.where(tmp71, tmp70, tmp69)
        tmp73 = tl.load(in_ptr2 + (tmp72 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp74 = tl.load(in_ptr3 + (tmp72 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp75 = tl.load(in_ptr4 + (tmp72 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.where(tmp18, tmp74, tmp75)
        tmp77 = tl.where(tmp13, tmp73, tmp76)
        tmp78 = tl.load(in_ptr5 + (tmp72 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tmp77 + tmp78
        tmp80 = tmp36 - tmp37
        tmp81 = tmp80 * tmp61
        tmp82 = tmp81 - tmp63
        tmp83 = tmp82 * tmp80
        tmp84 = tmp83 * tmp80
        tmp85 = tmp84 + tmp36
        tmp86 = tmp79 * tmp85
        tmp88 = tmp87 + tmp1
        tmp89 = tmp87 < 0
        tmp90 = tl.where(tmp89, tmp88, tmp87)
        tmp91 = tl.load(in_ptr2 + (tmp90 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.load(in_ptr3 + (tmp90 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = tl.load(in_ptr4 + (tmp90 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tl.where(tmp18, tmp92, tmp93)
        tmp95 = tl.where(tmp13, tmp91, tmp94)
        tmp96 = tl.load(in_ptr5 + (tmp90 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp97 = tmp95 + tmp96
        tmp98 = 2.0
        tmp99 = tmp98 - tmp37
        tmp100 = tmp99 * tmp39
        tmp101 = tmp100 - tmp41
        tmp102 = tmp101 * tmp99
        tmp103 = tmp102 + tmp44
        tmp104 = tmp103 * tmp99
        tmp105 = tmp104 - tmp47
        tmp106 = tmp97 * tmp105
        tmp108 = tmp107 + tmp1
        tmp109 = tmp107 < 0
        tmp110 = tl.where(tmp109, tmp108, tmp107)
        tmp111 = tl.load(in_ptr2 + (tmp8 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp112 = tl.load(in_ptr3 + (tmp8 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp113 = tl.load(in_ptr4 + (tmp8 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp114 = tl.where(tmp18, tmp112, tmp113)
        tmp115 = tl.where(tmp13, tmp111, tmp114)
        tmp116 = tl.load(in_ptr5 + (tmp8 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp117 = tmp115 + tmp116
        tmp118 = tmp117 * tmp48
        tmp119 = tl.load(in_ptr2 + (tmp53 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp120 = tl.load(in_ptr3 + (tmp53 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp121 = tl.load(in_ptr4 + (tmp53 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp122 = tl.where(tmp18, tmp120, tmp121)
        tmp123 = tl.where(tmp13, tmp119, tmp122)
        tmp124 = tl.load(in_ptr5 + (tmp53 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tmp123 + tmp124
        tmp126 = tmp125 * tmp67
        tmp127 = tl.load(in_ptr2 + (tmp72 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp128 = tl.load(in_ptr3 + (tmp72 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.load(in_ptr4 + (tmp72 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp130 = tl.where(tmp18, tmp128, tmp129)
        tmp131 = tl.where(tmp13, tmp127, tmp130)
        tmp132 = tl.load(in_ptr5 + (tmp72 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp133 = tmp131 + tmp132
        tmp134 = tmp133 * tmp85
        tmp135 = tl.load(in_ptr2 + (tmp90 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp136 = tl.load(in_ptr3 + (tmp90 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp137 = tl.load(in_ptr4 + (tmp90 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp138 = tl.where(tmp18, tmp136, tmp137)
        tmp139 = tl.where(tmp13, tmp135, tmp138)
        tmp140 = tl.load(in_ptr5 + (tmp90 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp141 = tmp139 + tmp140
        tmp142 = tmp141 * tmp105
        tmp144 = tmp143 + tmp1
        tmp145 = tmp143 < 0
        tmp146 = tl.where(tmp145, tmp144, tmp143)
        tmp147 = tl.load(in_ptr2 + (tmp8 + 8*tmp146 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp148 = tl.load(in_ptr3 + (tmp8 + 8*tmp146 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp149 = tl.load(in_ptr4 + (tmp8 + 8*tmp146 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.where(tmp18, tmp148, tmp149)
        tmp151 = tl.where(tmp13, tmp147, tmp150)
        tmp152 = tl.load(in_ptr5 + (tmp8 + 8*tmp146 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp153 = tmp151 + tmp152
        tmp154 = tmp153 * tmp48
        tmp155 = tl.load(in_ptr2 + (tmp53 + 8*tmp146 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp156 = tl.load(in_ptr3 + (tmp53 + 8*tmp146 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp157 = tl.load(in_ptr4 + (tmp53 + 8*tmp146 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp158 = tl.where(tmp18, tmp156, tmp157)
        tmp159 = tl.where(tmp13, tmp155, tmp158)
        tmp160 = tl.load(in_ptr5 + (tmp53 + 8*tmp146 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp161 = tmp159 + tmp160
        tmp162 = tmp161 * tmp67
        tmp163 = tl.load(in_ptr2 + (tmp72 + 8*tmp146 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp164 = tl.load(in_ptr3 + (tmp72 + 8*tmp146 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp165 = tl.load(in_ptr4 + (tmp72 + 8*tmp146 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp166 = tl.where(tmp18, tmp164, tmp165)
        tmp167 = tl.where(tmp13, tmp163, tmp166)
        tmp168 = tl.load(in_ptr5 + (tmp72 + 8*tmp146 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp169 * tmp85
        tmp171 = tl.load(in_ptr2 + (tmp90 + 8*tmp146 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp172 = tl.load(in_ptr3 + (tmp90 + 8*tmp146 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp173 = tl.load(in_ptr4 + (tmp90 + 8*tmp146 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp174 = tl.where(tmp18, tmp172, tmp173)
        tmp175 = tl.where(tmp13, tmp171, tmp174)
        tmp176 = tl.load(in_ptr5 + (tmp90 + 8*tmp146 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp177 = tmp175 + tmp176
        tmp178 = tmp177 * tmp105
        tmp180 = tmp179 + tmp1
        tmp181 = tmp179 < 0
        tmp182 = tl.where(tmp181, tmp180, tmp179)
        tmp183 = tl.load(in_ptr2 + (tmp8 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp184 = tl.load(in_ptr3 + (tmp8 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp185 = tl.load(in_ptr4 + (tmp8 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp186 = tl.where(tmp18, tmp184, tmp185)
        tmp187 = tl.where(tmp13, tmp183, tmp186)
        tmp188 = tl.load(in_ptr5 + (tmp8 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp189 = tmp187 + tmp188
        tmp190 = tmp189 * tmp48
        tmp191 = tl.load(in_ptr2 + (tmp53 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp192 = tl.load(in_ptr3 + (tmp53 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp193 = tl.load(in_ptr4 + (tmp53 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp194 = tl.where(tmp18, tmp192, tmp193)
        tmp195 = tl.where(tmp13, tmp191, tmp194)
        tmp196 = tl.load(in_ptr5 + (tmp53 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp197 = tmp195 + tmp196
        tmp198 = tmp197 * tmp67
        tmp199 = tl.load(in_ptr2 + (tmp72 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp200 = tl.load(in_ptr3 + (tmp72 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp201 = tl.load(in_ptr4 + (tmp72 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp202 = tl.where(tmp18, tmp200, tmp201)
        tmp203 = tl.where(tmp13, tmp199, tmp202)
        tmp204 = tl.load(in_ptr5 + (tmp72 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp205 = tmp203 + tmp204
        tmp206 = tmp205 * tmp85
        tmp207 = tl.load(in_ptr2 + (tmp90 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp208 = tl.load(in_ptr3 + (tmp90 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp209 = tl.load(in_ptr4 + (tmp90 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp210 = tl.where(tmp18, tmp208, tmp209)
        tmp211 = tl.where(tmp13, tmp207, tmp210)
        tmp212 = tl.load(in_ptr5 + (tmp90 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp213 = tmp211 + tmp212
        tmp214 = tmp213 * tmp105
        tmp215 = tmp49 + tmp68
        tmp216 = tmp215 + tmp86
        tmp217 = tmp216 + tmp106
        tmp218 = r3
        tmp219 = tmp218.to(tl.float32)
        tmp220 = tmp219 * tmp30
        tmp221 = libdevice.floor(tmp220)
        tmp222 = tmp220 - tmp221
        tmp223 = triton_helpers.maximum(tmp222, tmp34)
        tmp224 = triton_helpers.minimum(tmp223, tmp36)
        tmp225 = tmp224 + tmp36
        tmp226 = tmp225 * tmp39
        tmp227 = tmp226 - tmp41
        tmp228 = tmp227 * tmp225
        tmp229 = tmp228 + tmp44
        tmp230 = tmp229 * tmp225
        tmp231 = tmp230 - tmp47
        tmp232 = tmp217 * tmp231
        tmp233 = tmp118 + tmp126
        tmp234 = tmp233 + tmp134
        tmp235 = tmp234 + tmp142
        tmp236 = tmp224 * tmp61
        tmp237 = tmp236 - tmp63
        tmp238 = tmp237 * tmp224
        tmp239 = tmp238 * tmp224
        tmp240 = tmp239 + tmp36
        tmp241 = tmp235 * tmp240
        tmp242 = tmp232 + tmp241
        tmp243 = tmp154 + tmp162
        tmp244 = tmp243 + tmp170
        tmp245 = tmp244 + tmp178
        tmp246 = tmp36 - tmp224
        tmp247 = tmp246 * tmp61
        tmp248 = tmp247 - tmp63
        tmp249 = tmp248 * tmp246
        tmp250 = tmp249 * tmp246
        tmp251 = tmp250 + tmp36
        tmp252 = tmp245 * tmp251
        tmp253 = tmp242 + tmp252
        tmp254 = tmp190 + tmp198
        tmp255 = tmp254 + tmp206
        tmp256 = tmp255 + tmp214
        tmp257 = tmp98 - tmp224
        tmp258 = tmp257 * tmp39
        tmp259 = tmp258 - tmp41
        tmp260 = tmp259 * tmp257
        tmp261 = tmp260 + tmp44
        tmp262 = tmp261 * tmp257
        tmp263 = tmp262 - tmp47
        tmp264 = tmp256 * tmp263
        tmp265 = tmp253 + tmp264
        tmp266 = tl.load(in_ptr12 + (r6 + 256*(r4 + 8*x0) + 32768*x1), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp267 = tl.load(in_ptr13 + (r6 + 256*((-128) + r4 + 8*x0) + 16384*x1), rmask & tmp18 & xmask, eviction_policy='evict_first', other=0.0)
        tmp268 = tl.load(in_ptr14 + (r6 + 256*((-192) + r4 + 8*x0) + 16384*x1), rmask & tmp20 & xmask, eviction_policy='evict_first', other=0.0)
        tmp269 = tl.where(tmp18, tmp267, tmp268)
        tmp270 = tl.where(tmp13, tmp266, tmp269)
        tmp272 = tmp270 + tmp271
        tmp273 = tmp272 + tmp265
        tmp274 = tl.broadcast_to(tmp273, [XBLOCK, RBLOCK])
        tmp275_mean_next, tmp275_m2_next, tmp275_weight_next = triton_helpers.welford_reduce(
            tmp274, tmp275_mean, tmp275_m2, tmp275_weight, roffset == 0
        )
        tmp275_mean = tl.where(rmask & xmask, tmp275_mean_next, tmp275_mean)
        tmp275_m2 = tl.where(rmask & xmask, tmp275_m2_next, tmp275_m2)
        tmp275_weight = tl.where(rmask & xmask, tmp275_weight_next, tmp275_weight)
        tl.store(in_out_ptr0 + (r7 + 2048*x5), tmp273, rmask & xmask)
    tmp275_tmp, tmp276_tmp, tmp277_tmp = triton_helpers.welford(
        tmp275_mean, tmp275_m2, tmp275_weight, 1
    )
    tmp275 = tmp275_tmp[:, None]
    tmp276 = tmp276_tmp[:, None]
    tmp277 = tmp277_tmp[:, None]
    tl.store(out_ptr15 + (x5), tmp275, xmask)
    tl.store(out_ptr16 + (x5), tmp276, xmask)
    tmp278 = 2048.0
    tmp279 = tmp276 / tmp278
    tmp280 = 1e-05
    tmp281 = tmp279 + tmp280
    tmp282 = libdevice.rsqrt(tmp281)
    tl.store(out_ptr17 + (x5), tmp282, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g5/cg5p6usnn67yegz4k7t533kl4nn4gkrivsw7ttwoqtxrfrgxwkto.py
# Topologically Sorted Source Nodes: [out3_53, out3_54], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   out3_53 => cat_10
#   out3_54 => add_142
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_33, %convolution_34, %convolution_35], 1), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_10, %add_135), kwargs = {})
triton_poi_fused_add_cat_43 = async_compile.triton('triton_poi_fused_add_cat_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x2 = xindex // 65536
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 32768*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-128) + x1) + 16384*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 256*((-192) + x1) + 16384*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/cw/ccwihgs56ywsbiovvqbxtxfcfitg4plywktdtr232bvien23j5if.py
# Topologically Sorted Source Nodes: [conv2d_36, group_norm_36], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   conv2d_36 => convolution_36
#   group_norm_36 => add_143, rsqrt_36, var_mean_36
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_142, %primals_111, %primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-05), kwargs = {})
#   %rsqrt_36 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_143,), kwargs = {})
triton_red_fused_convolution_native_group_norm_44 = async_compile.triton('triton_red_fused_convolution_native_group_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_44(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tmp7 = 2048.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cync7qh7es5oldkvjcyhnwyrvlilvbmstbtszk7mrl7crcl2l65k.py
# Topologically Sorted Source Nodes: [tmp_out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   tmp_out => convolution_37
# Graph fragment:
#   %convolution_37 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_36, %primals_115, %primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_45 = async_compile.triton('triton_poi_fused_convolution_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_45(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ip/cipvjs7kyfw5f7j6piyh75iwhxqxkhkui6d7j25pby4f3lnvmpqh.py
# Topologically Sorted Source Nodes: [ll_1, tmp_out_, add_2, previous, out1_33], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_2 => add_145
#   ll_1 => convolution_38
#   out1_33 => add_147, rsqrt_37, var_mean_37
#   previous => add_146
#   tmp_out_ => convolution_39
# Graph fragment:
#   %convolution_38 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_36, %primals_117, %primals_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_39 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_37, %primals_119, %primals_120, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %convolution_38), kwargs = {})
#   %add_146 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %convolution_39), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_74, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_74, 1e-05), kwargs = {})
#   %rsqrt_37 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_147,), kwargs = {})
triton_red_fused_add_convolution_native_group_norm_46 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
    tl.store(out_ptr1 + (x4), tmp11, xmask)
    tmp13 = 2048.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr2 + (x4), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7i/c7iieddmzkucbx2qbcm4uukxfeiphflqe63pi4nzfdzncpeoaie3.py
# Topologically Sorted Source Nodes: [up2, out3_68, out3_69, out3_83, out3_84, up2_2, low2_1, out1_51], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   low2_1 => add_218
#   out1_51 => add_219, rsqrt_55, var_mean_55
#   out3_68 => cat_13
#   out3_69 => add_167
#   out3_83 => cat_16
#   out3_84 => add_188
#   up2 => add_73, add_74, add_75, add_76, add_77, add_78, add_79, add_80, add_81, add_82, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type, floor, floor_1, iota, mul_60, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, sub_30, sub_31, sub_34, sub_35, sub_36, sub_37, sub_38, sub_39, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45, sub_46, sub_47, sub_48, sub_49
#   up2_2 => _unsafe_index_32, _unsafe_index_33, _unsafe_index_34, _unsafe_index_35, _unsafe_index_36, _unsafe_index_37, _unsafe_index_38, _unsafe_index_39, _unsafe_index_40, _unsafe_index_41, _unsafe_index_42, _unsafe_index_43, _unsafe_index_44, _unsafe_index_45, _unsafe_index_46, _unsafe_index_47, add_203, add_204, add_205, add_206, add_207, add_208, add_209, add_210, add_211, add_212, add_213, add_214, add_215, add_216, add_217, mul_228, mul_229, mul_230, mul_231, mul_232, mul_233, mul_234, mul_235, mul_236, mul_237, mul_238, mul_239, mul_240, mul_241, mul_242, mul_243, mul_244, mul_245, mul_246, mul_247
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_60 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.42857142857142855), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_60,), kwargs = {})
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_180,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_180, %floor_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_30, 0.0), kwargs = {})
#   %clamp_max : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_60, %floor), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_31, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1.0), kwargs = {})
#   %add_73 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, 1.0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, -0.75), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_62, -3.75), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %add_73), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, -6.0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %add_73), kwargs = {})
#   %sub_35 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_64, -3.0), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 1.25), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_65, 2.25), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_1), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %clamp_max_1), kwargs = {})
#   %add_75 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, 1), kwargs = {})
#   %sub_37 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_1), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, 1.25), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_68, 2.25), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %sub_37), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %sub_37), kwargs = {})
#   %add_76 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, 1), kwargs = {})
#   %sub_39 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_1), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, -0.75), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_71, -3.75), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %sub_39), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, -6.0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, %sub_39), kwargs = {})
#   %sub_41 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_73, -3.0), kwargs = {})
#   %add_78 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, 1.0), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, -0.75), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_74, -3.75), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %add_78), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, -6.0), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %add_78), kwargs = {})
#   %sub_43 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_76, -3.0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 1.25), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_77, 2.25), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %clamp_max), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %clamp_max), kwargs = {})
#   %add_80 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, 1), kwargs = {})
#   %sub_45 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, 1.25), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_80, 2.25), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %sub_45), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %sub_45), kwargs = {})
#   %add_81 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, 1), kwargs = {})
#   %sub_47 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, -0.75), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, -3.75), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %sub_47), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, -6.0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %sub_47), kwargs = {})
#   %sub_49 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_85, -3.0), kwargs = {})
#   %cat_13 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_46, %convolution_47, %convolution_48], 1), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_13, %add_160), kwargs = {})
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_55, %convolution_56, %convolution_57], 1), kwargs = {})
#   %add_188 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_16, %add_181), kwargs = {})
#   %_unsafe_index_32 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_2, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_33 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_2, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_34 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_2, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_35 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_2, %clamp_max_9]), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_32, %sub_35), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_33, %add_75), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_228, %mul_229), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_34, %add_76), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_203, %mul_230), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_35, %sub_41), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_204, %mul_231), kwargs = {})
#   %_unsafe_index_36 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_10, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_37 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_10, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_38 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_10, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_39 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_10, %clamp_max_9]), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_36, %sub_35), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_37, %add_75), kwargs = {})
#   %add_206 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %mul_233), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_38, %add_76), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_206, %mul_234), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_39, %sub_41), kwargs = {})
#   %add_208 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_207, %mul_235), kwargs = {})
#   %_unsafe_index_40 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_18, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_41 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_18, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_42 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_18, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_43 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_18, %clamp_max_9]), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_40, %sub_35), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_41, %add_75), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %mul_237), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_42, %add_76), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_209, %mul_238), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_43, %sub_41), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_210, %mul_239), kwargs = {})
#   %_unsafe_index_44 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_26, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_45 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_26, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_46 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_26, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_47 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_188, [None, None, %clamp_max_26, %clamp_max_9]), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_44, %sub_35), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_45, %add_75), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_240, %mul_241), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_46, %add_76), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_212, %mul_242), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_47, %sub_41), kwargs = {})
#   %add_214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_213, %mul_243), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %sub_43), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_208, %add_80), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_244, %mul_245), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, %add_81), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_215, %mul_246), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_214, %sub_49), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_216, %mul_247), kwargs = {})
#   %add_218 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %add_217), kwargs = {})
#   %var_mean_55 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_110, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_110, 1e-05), kwargs = {})
#   %rsqrt_55 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_219,), kwargs = {})
triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr15, out_ptr16, out_ptr17, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x5 = xindex
    tmp275_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = ((rindex // 8) % 8)
        r2 = (rindex % 8)
        r4 = rindex // 64
        r7 = rindex
        r6 = (rindex % 64)
        tmp0 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp87 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp107 = tl.load(in_ptr9 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp135 = tl.load(in_ptr10 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp179 = tl.load(in_ptr11 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp271 = tl.load(in_ptr15 + (r7 + 512*x5), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp9 = r4 + 8*x0
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tl.full([1, 1], 128, tl.int64)
        tmp13 = tmp9 < tmp12
        tmp14 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp9 >= tmp12
        tmp16 = tl.full([1, 1], 192, tl.int64)
        tmp17 = tmp9 < tmp16
        tmp18 = tmp15 & tmp17
        tmp19 = tl.load(in_ptr3 + (tmp8 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp9 >= tmp16
        tmp21 = tl.full([1, 1], 256, tl.int64)
        tmp22 = tmp9 < tmp21
        tmp23 = tl.load(in_ptr4 + (tmp8 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.where(tmp18, tmp19, tmp23)
        tmp25 = tl.where(tmp13, tmp14, tmp24)
        tmp26 = tl.load(in_ptr5 + (tmp8 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 + tmp26
        tmp28 = r2
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 0.42857142857142855
        tmp31 = tmp29 * tmp30
        tmp32 = libdevice.floor(tmp31)
        tmp33 = tmp31 - tmp32
        tmp34 = 0.0
        tmp35 = triton_helpers.maximum(tmp33, tmp34)
        tmp36 = 1.0
        tmp37 = triton_helpers.minimum(tmp35, tmp36)
        tmp38 = tmp37 + tmp36
        tmp39 = -0.75
        tmp40 = tmp38 * tmp39
        tmp41 = -3.75
        tmp42 = tmp40 - tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = -6.0
        tmp45 = tmp43 + tmp44
        tmp46 = tmp45 * tmp38
        tmp47 = -3.0
        tmp48 = tmp46 - tmp47
        tmp49 = tmp27 * tmp48
        tmp51 = tmp50 + tmp1
        tmp52 = tmp50 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp50)
        tmp54 = tl.load(in_ptr2 + (tmp53 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr3 + (tmp53 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr4 + (tmp53 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.where(tmp18, tmp55, tmp56)
        tmp58 = tl.where(tmp13, tmp54, tmp57)
        tmp59 = tl.load(in_ptr5 + (tmp53 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp58 + tmp59
        tmp61 = 1.25
        tmp62 = tmp37 * tmp61
        tmp63 = 2.25
        tmp64 = tmp62 - tmp63
        tmp65 = tmp64 * tmp37
        tmp66 = tmp65 * tmp37
        tmp67 = tmp66 + tmp36
        tmp68 = tmp60 * tmp67
        tmp70 = tmp69 + tmp1
        tmp71 = tmp69 < 0
        tmp72 = tl.where(tmp71, tmp70, tmp69)
        tmp73 = tl.load(in_ptr2 + (tmp72 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp74 = tl.load(in_ptr3 + (tmp72 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp75 = tl.load(in_ptr4 + (tmp72 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.where(tmp18, tmp74, tmp75)
        tmp77 = tl.where(tmp13, tmp73, tmp76)
        tmp78 = tl.load(in_ptr5 + (tmp72 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tmp77 + tmp78
        tmp80 = tmp36 - tmp37
        tmp81 = tmp80 * tmp61
        tmp82 = tmp81 - tmp63
        tmp83 = tmp82 * tmp80
        tmp84 = tmp83 * tmp80
        tmp85 = tmp84 + tmp36
        tmp86 = tmp79 * tmp85
        tmp88 = tmp87 + tmp1
        tmp89 = tmp87 < 0
        tmp90 = tl.where(tmp89, tmp88, tmp87)
        tmp91 = tl.load(in_ptr2 + (tmp90 + 4*tmp4 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.load(in_ptr3 + (tmp90 + 4*tmp4 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = tl.load(in_ptr4 + (tmp90 + 4*tmp4 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tl.where(tmp18, tmp92, tmp93)
        tmp95 = tl.where(tmp13, tmp91, tmp94)
        tmp96 = tl.load(in_ptr5 + (tmp90 + 4*tmp4 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp97 = tmp95 + tmp96
        tmp98 = 2.0
        tmp99 = tmp98 - tmp37
        tmp100 = tmp99 * tmp39
        tmp101 = tmp100 - tmp41
        tmp102 = tmp101 * tmp99
        tmp103 = tmp102 + tmp44
        tmp104 = tmp103 * tmp99
        tmp105 = tmp104 - tmp47
        tmp106 = tmp97 * tmp105
        tmp108 = tmp107 + tmp1
        tmp109 = tmp107 < 0
        tmp110 = tl.where(tmp109, tmp108, tmp107)
        tmp111 = tl.load(in_ptr2 + (tmp8 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp112 = tl.load(in_ptr3 + (tmp8 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp113 = tl.load(in_ptr4 + (tmp8 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp114 = tl.where(tmp18, tmp112, tmp113)
        tmp115 = tl.where(tmp13, tmp111, tmp114)
        tmp116 = tl.load(in_ptr5 + (tmp8 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp117 = tmp115 + tmp116
        tmp118 = tmp117 * tmp48
        tmp119 = tl.load(in_ptr2 + (tmp53 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp120 = tl.load(in_ptr3 + (tmp53 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp121 = tl.load(in_ptr4 + (tmp53 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp122 = tl.where(tmp18, tmp120, tmp121)
        tmp123 = tl.where(tmp13, tmp119, tmp122)
        tmp124 = tl.load(in_ptr5 + (tmp53 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tmp123 + tmp124
        tmp126 = tmp125 * tmp67
        tmp127 = tl.load(in_ptr2 + (tmp72 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp128 = tl.load(in_ptr3 + (tmp72 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.load(in_ptr4 + (tmp72 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp130 = tl.where(tmp18, tmp128, tmp129)
        tmp131 = tl.where(tmp13, tmp127, tmp130)
        tmp132 = tl.load(in_ptr5 + (tmp72 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp133 = tmp131 + tmp132
        tmp134 = tmp133 * tmp85
        tmp136 = tmp135 + tmp1
        tmp137 = tmp135 < 0
        tmp138 = tl.where(tmp137, tmp136, tmp135)
        tmp139 = tl.load(in_ptr2 + (tmp8 + 4*tmp138 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp140 = tl.load(in_ptr3 + (tmp8 + 4*tmp138 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp141 = tl.load(in_ptr4 + (tmp8 + 4*tmp138 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp142 = tl.where(tmp18, tmp140, tmp141)
        tmp143 = tl.where(tmp13, tmp139, tmp142)
        tmp144 = tl.load(in_ptr5 + (tmp8 + 4*tmp138 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp145 = tmp143 + tmp144
        tmp146 = tmp145 * tmp48
        tmp147 = tl.load(in_ptr2 + (tmp90 + 4*tmp110 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp148 = tl.load(in_ptr3 + (tmp90 + 4*tmp110 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp149 = tl.load(in_ptr4 + (tmp90 + 4*tmp110 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.where(tmp18, tmp148, tmp149)
        tmp151 = tl.where(tmp13, tmp147, tmp150)
        tmp152 = tl.load(in_ptr5 + (tmp90 + 4*tmp110 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp153 = tmp151 + tmp152
        tmp154 = tmp153 * tmp105
        tmp155 = tl.load(in_ptr2 + (tmp53 + 4*tmp138 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp156 = tl.load(in_ptr3 + (tmp53 + 4*tmp138 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp157 = tl.load(in_ptr4 + (tmp53 + 4*tmp138 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp158 = tl.where(tmp18, tmp156, tmp157)
        tmp159 = tl.where(tmp13, tmp155, tmp158)
        tmp160 = tl.load(in_ptr5 + (tmp53 + 4*tmp138 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp161 = tmp159 + tmp160
        tmp162 = tmp161 * tmp67
        tmp163 = tl.load(in_ptr2 + (tmp72 + 4*tmp138 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp164 = tl.load(in_ptr3 + (tmp72 + 4*tmp138 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp165 = tl.load(in_ptr4 + (tmp72 + 4*tmp138 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp166 = tl.where(tmp18, tmp164, tmp165)
        tmp167 = tl.where(tmp13, tmp163, tmp166)
        tmp168 = tl.load(in_ptr5 + (tmp72 + 4*tmp138 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp169 * tmp85
        tmp171 = tl.load(in_ptr2 + (tmp90 + 4*tmp138 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp172 = tl.load(in_ptr3 + (tmp90 + 4*tmp138 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp173 = tl.load(in_ptr4 + (tmp90 + 4*tmp138 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp174 = tl.where(tmp18, tmp172, tmp173)
        tmp175 = tl.where(tmp13, tmp171, tmp174)
        tmp176 = tl.load(in_ptr5 + (tmp90 + 4*tmp138 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp177 = tmp175 + tmp176
        tmp178 = tmp177 * tmp105
        tmp180 = tmp179 + tmp1
        tmp181 = tmp179 < 0
        tmp182 = tl.where(tmp181, tmp180, tmp179)
        tmp183 = tl.load(in_ptr2 + (tmp8 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp184 = tl.load(in_ptr3 + (tmp8 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp185 = tl.load(in_ptr4 + (tmp8 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp186 = tl.where(tmp18, tmp184, tmp185)
        tmp187 = tl.where(tmp13, tmp183, tmp186)
        tmp188 = tl.load(in_ptr5 + (tmp8 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp189 = tmp187 + tmp188
        tmp190 = tmp189 * tmp48
        tmp191 = tl.load(in_ptr2 + (tmp53 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp192 = tl.load(in_ptr3 + (tmp53 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp193 = tl.load(in_ptr4 + (tmp53 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp194 = tl.where(tmp18, tmp192, tmp193)
        tmp195 = tl.where(tmp13, tmp191, tmp194)
        tmp196 = tl.load(in_ptr5 + (tmp53 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp197 = tmp195 + tmp196
        tmp198 = tmp197 * tmp67
        tmp199 = tl.load(in_ptr2 + (tmp72 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp200 = tl.load(in_ptr3 + (tmp72 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp201 = tl.load(in_ptr4 + (tmp72 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp202 = tl.where(tmp18, tmp200, tmp201)
        tmp203 = tl.where(tmp13, tmp199, tmp202)
        tmp204 = tl.load(in_ptr5 + (tmp72 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp205 = tmp203 + tmp204
        tmp206 = tmp205 * tmp85
        tmp207 = tl.load(in_ptr2 + (tmp90 + 4*tmp182 + 16*(r4 + 8*x0) + 2048*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp208 = tl.load(in_ptr3 + (tmp90 + 4*tmp182 + 16*((-128) + r4 + 8*x0) + 1024*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp209 = tl.load(in_ptr4 + (tmp90 + 4*tmp182 + 16*((-192) + r4 + 8*x0) + 1024*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp210 = tl.where(tmp18, tmp208, tmp209)
        tmp211 = tl.where(tmp13, tmp207, tmp210)
        tmp212 = tl.load(in_ptr5 + (tmp90 + 4*tmp182 + 16*r4 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp213 = tmp211 + tmp212
        tmp214 = tmp213 * tmp105
        tmp215 = tmp49 + tmp68
        tmp216 = tmp215 + tmp86
        tmp217 = tmp216 + tmp106
        tmp218 = r3
        tmp219 = tmp218.to(tl.float32)
        tmp220 = tmp219 * tmp30
        tmp221 = libdevice.floor(tmp220)
        tmp222 = tmp220 - tmp221
        tmp223 = triton_helpers.maximum(tmp222, tmp34)
        tmp224 = triton_helpers.minimum(tmp223, tmp36)
        tmp225 = tmp224 + tmp36
        tmp226 = tmp225 * tmp39
        tmp227 = tmp226 - tmp41
        tmp228 = tmp227 * tmp225
        tmp229 = tmp228 + tmp44
        tmp230 = tmp229 * tmp225
        tmp231 = tmp230 - tmp47
        tmp232 = tmp217 * tmp231
        tmp233 = tmp118 + tmp126
        tmp234 = tmp233 + tmp134
        tmp235 = tmp234 + tmp154
        tmp236 = tmp224 * tmp61
        tmp237 = tmp236 - tmp63
        tmp238 = tmp237 * tmp224
        tmp239 = tmp238 * tmp224
        tmp240 = tmp239 + tmp36
        tmp241 = tmp235 * tmp240
        tmp242 = tmp232 + tmp241
        tmp243 = tmp146 + tmp162
        tmp244 = tmp243 + tmp170
        tmp245 = tmp244 + tmp178
        tmp246 = tmp36 - tmp224
        tmp247 = tmp246 * tmp61
        tmp248 = tmp247 - tmp63
        tmp249 = tmp248 * tmp246
        tmp250 = tmp249 * tmp246
        tmp251 = tmp250 + tmp36
        tmp252 = tmp245 * tmp251
        tmp253 = tmp242 + tmp252
        tmp254 = tmp190 + tmp198
        tmp255 = tmp254 + tmp206
        tmp256 = tmp255 + tmp214
        tmp257 = tmp98 - tmp224
        tmp258 = tmp257 * tmp39
        tmp259 = tmp258 - tmp41
        tmp260 = tmp259 * tmp257
        tmp261 = tmp260 + tmp44
        tmp262 = tmp261 * tmp257
        tmp263 = tmp262 - tmp47
        tmp264 = tmp256 * tmp263
        tmp265 = tmp253 + tmp264
        tmp266 = tl.load(in_ptr12 + (r6 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp267 = tl.load(in_ptr13 + (r6 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_first', other=0.0)
        tmp268 = tl.load(in_ptr14 + (r6 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_first', other=0.0)
        tmp269 = tl.where(tmp18, tmp267, tmp268)
        tmp270 = tl.where(tmp13, tmp266, tmp269)
        tmp272 = tmp270 + tmp271
        tmp273 = tmp272 + tmp265
        tmp274 = tl.broadcast_to(tmp273, [XBLOCK, RBLOCK])
        tmp275_mean_next, tmp275_m2_next, tmp275_weight_next = triton_helpers.welford_reduce(
            tmp274, tmp275_mean, tmp275_m2, tmp275_weight, roffset == 0
        )
        tmp275_mean = tl.where(rmask & xmask, tmp275_mean_next, tmp275_mean)
        tmp275_m2 = tl.where(rmask & xmask, tmp275_m2_next, tmp275_m2)
        tmp275_weight = tl.where(rmask & xmask, tmp275_weight_next, tmp275_weight)
        tl.store(in_out_ptr0 + (r7 + 512*x5), tmp273, rmask & xmask)
    tmp275_tmp, tmp276_tmp, tmp277_tmp = triton_helpers.welford(
        tmp275_mean, tmp275_m2, tmp275_weight, 1
    )
    tmp275 = tmp275_tmp[:, None]
    tmp276 = tmp276_tmp[:, None]
    tmp277 = tmp277_tmp[:, None]
    tl.store(out_ptr15 + (x5), tmp275, xmask)
    tl.store(out_ptr16 + (x5), tmp276, xmask)
    tmp278 = 512.0
    tmp279 = tmp276 / tmp278
    tmp280 = 1e-05
    tmp281 = tmp279 + tmp280
    tmp282 = libdevice.rsqrt(tmp281)
    tl.store(out_ptr17 + (x5), tmp282, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g5/cg5gooxa3dnl7opddv3yhhzlk4uk3refyzmnsprkyp4jcunxz7d7.py
# Topologically Sorted Source Nodes: [up2_1, out3_58, out3_59, out3_88, out3_89, up2_3, hg_1, out1_54], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   hg_1 => add_255
#   out1_54 => add_256, rsqrt_58, var_mean_58
#   out3_58 => cat_11
#   out3_59 => add_153
#   out3_88 => cat_17
#   out3_89 => add_225
#   up2_1 => add_110, add_111, add_112, add_113, add_114, add_115, add_116, add_117, add_118, add_119, clamp_max_34, clamp_max_35, clamp_min_34, clamp_min_35, convert_element_type_4, floor_2, floor_3, iota_2, mul_112, mul_114, mul_115, mul_116, mul_117, mul_118, mul_119, mul_120, mul_121, mul_122, mul_123, mul_124, mul_125, mul_126, mul_127, mul_128, mul_129, mul_130, mul_131, mul_132, mul_133, mul_134, mul_135, mul_136, mul_137, sub_53, sub_54, sub_57, sub_58, sub_59, sub_60, sub_61, sub_62, sub_63, sub_64, sub_65, sub_66, sub_67, sub_68, sub_69, sub_70, sub_71, sub_72
#   up2_3 => _unsafe_index_48, _unsafe_index_49, _unsafe_index_50, _unsafe_index_51, _unsafe_index_52, _unsafe_index_53, _unsafe_index_54, _unsafe_index_55, _unsafe_index_56, _unsafe_index_57, _unsafe_index_58, _unsafe_index_59, _unsafe_index_60, _unsafe_index_61, _unsafe_index_62, _unsafe_index_63, add_240, add_241, add_242, add_243, add_244, add_245, add_246, add_247, add_248, add_249, add_250, add_251, add_252, add_253, add_254, mul_280, mul_281, mul_282, mul_283, mul_284, mul_285, mul_286, mul_287, mul_288, mul_289, mul_290, mul_291, mul_292, mul_293, mul_294, mul_295, mul_296, mul_297, mul_298, mul_299
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.4666666666666667), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_112,), kwargs = {})
#   %floor_3 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_199,), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_199, %floor_3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_53, 0.0), kwargs = {})
#   %clamp_max_34 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 1.0), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_112, %floor_2), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_54, 0.0), kwargs = {})
#   %clamp_max_35 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 1.0), kwargs = {})
#   %add_110 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_35, 1.0), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, -0.75), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_114, -3.75), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %add_110), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, -6.0), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_111, %add_110), kwargs = {})
#   %sub_58 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_116, -3.0), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_35, 1.25), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_117, 2.25), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %clamp_max_35), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %clamp_max_35), kwargs = {})
#   %add_112 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, 1), kwargs = {})
#   %sub_60 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_35), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, 1.25), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_120, 2.25), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %sub_60), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %sub_60), kwargs = {})
#   %add_113 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, 1), kwargs = {})
#   %sub_62 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_35), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, -0.75), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_123, -3.75), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %sub_62), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, -6.0), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %sub_62), kwargs = {})
#   %sub_64 : [num_users=16] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_125, -3.0), kwargs = {})
#   %add_115 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_34, 1.0), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, -0.75), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_126, -3.75), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %add_115), kwargs = {})
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, -6.0), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %add_115), kwargs = {})
#   %sub_66 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_128, -3.0), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_34, 1.25), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_129, 2.25), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %clamp_max_34), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %clamp_max_34), kwargs = {})
#   %add_117 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, 1), kwargs = {})
#   %sub_68 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_34), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, 1.25), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_132, 2.25), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %sub_68), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %sub_68), kwargs = {})
#   %add_118 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, 1), kwargs = {})
#   %sub_70 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_34), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, -0.75), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_135, -3.75), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %sub_70), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, -6.0), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %sub_70), kwargs = {})
#   %sub_72 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_137, -3.0), kwargs = {})
#   %cat_11 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_40, %convolution_41, %convolution_42], 1), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_11, %add_146), kwargs = {})
#   %cat_17 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_58, %convolution_59, %convolution_60], 1), kwargs = {})
#   %add_225 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_17, %add_218), kwargs = {})
#   %_unsafe_index_48 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_36, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_49 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_36, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_50 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_36, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_51 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_36, %clamp_max_43]), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_48, %sub_58), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_49, %add_112), kwargs = {})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_280, %mul_281), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_50, %add_113), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_240, %mul_282), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_51, %sub_64), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_241, %mul_283), kwargs = {})
#   %_unsafe_index_52 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_44, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_53 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_44, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_54 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_44, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_55 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_44, %clamp_max_43]), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_52, %sub_58), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_53, %add_112), kwargs = {})
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %mul_285), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_54, %add_113), kwargs = {})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_243, %mul_286), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_55, %sub_64), kwargs = {})
#   %add_245 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_244, %mul_287), kwargs = {})
#   %_unsafe_index_56 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_52, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_57 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_52, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_58 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_52, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_59 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_52, %clamp_max_43]), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_56, %sub_58), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_57, %add_112), kwargs = {})
#   %add_246 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_288, %mul_289), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_58, %add_113), kwargs = {})
#   %add_247 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_246, %mul_290), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_59, %sub_64), kwargs = {})
#   %add_248 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_247, %mul_291), kwargs = {})
#   %_unsafe_index_60 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_60, %clamp_max_37]), kwargs = {})
#   %_unsafe_index_61 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_60, %clamp_max_39]), kwargs = {})
#   %_unsafe_index_62 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_60, %clamp_max_41]), kwargs = {})
#   %_unsafe_index_63 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_225, [None, None, %clamp_max_60, %clamp_max_43]), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_60, %sub_58), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_61, %add_112), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %mul_293), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_62, %add_113), kwargs = {})
#   %add_250 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_249, %mul_294), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_63, %sub_64), kwargs = {})
#   %add_251 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_250, %mul_295), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_242, %sub_66), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_245, %add_117), kwargs = {})
#   %add_252 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %mul_297), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_248, %add_118), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_252, %mul_298), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_251, %sub_72), kwargs = {})
#   %add_254 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_253, %mul_299), kwargs = {})
#   %add_255 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %add_254), kwargs = {})
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_116, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_256 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_116, 1e-05), kwargs = {})
#   %rsqrt_58 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_256,), kwargs = {})
triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr15, out_ptr16, out_ptr17, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x5 = xindex
    tmp275_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp275_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = ((rindex // 16) % 16)
        r2 = (rindex % 16)
        r4 = rindex // 256
        r7 = rindex
        r6 = (rindex % 256)
        tmp0 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp87 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp107 = tl.load(in_ptr9 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp135 = tl.load(in_ptr10 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp179 = tl.load(in_ptr11 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp271 = tl.load(in_ptr15 + (r7 + 2048*x5), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp9 = r4 + 8*x0
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tl.full([1, 1], 128, tl.int64)
        tmp13 = tmp9 < tmp12
        tmp14 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp9 >= tmp12
        tmp16 = tl.full([1, 1], 192, tl.int64)
        tmp17 = tmp9 < tmp16
        tmp18 = tmp15 & tmp17
        tmp19 = tl.load(in_ptr3 + (tmp8 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp9 >= tmp16
        tmp21 = tl.full([1, 1], 256, tl.int64)
        tmp22 = tmp9 < tmp21
        tmp23 = tl.load(in_ptr4 + (tmp8 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.where(tmp18, tmp19, tmp23)
        tmp25 = tl.where(tmp13, tmp14, tmp24)
        tmp26 = tl.load(in_ptr5 + (tmp8 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 + tmp26
        tmp28 = r2
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 0.4666666666666667
        tmp31 = tmp29 * tmp30
        tmp32 = libdevice.floor(tmp31)
        tmp33 = tmp31 - tmp32
        tmp34 = 0.0
        tmp35 = triton_helpers.maximum(tmp33, tmp34)
        tmp36 = 1.0
        tmp37 = triton_helpers.minimum(tmp35, tmp36)
        tmp38 = tmp37 + tmp36
        tmp39 = -0.75
        tmp40 = tmp38 * tmp39
        tmp41 = -3.75
        tmp42 = tmp40 - tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = -6.0
        tmp45 = tmp43 + tmp44
        tmp46 = tmp45 * tmp38
        tmp47 = -3.0
        tmp48 = tmp46 - tmp47
        tmp49 = tmp27 * tmp48
        tmp51 = tmp50 + tmp1
        tmp52 = tmp50 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp50)
        tmp54 = tl.load(in_ptr2 + (tmp53 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr3 + (tmp53 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr4 + (tmp53 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.where(tmp18, tmp55, tmp56)
        tmp58 = tl.where(tmp13, tmp54, tmp57)
        tmp59 = tl.load(in_ptr5 + (tmp53 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp58 + tmp59
        tmp61 = 1.25
        tmp62 = tmp37 * tmp61
        tmp63 = 2.25
        tmp64 = tmp62 - tmp63
        tmp65 = tmp64 * tmp37
        tmp66 = tmp65 * tmp37
        tmp67 = tmp66 + tmp36
        tmp68 = tmp60 * tmp67
        tmp70 = tmp69 + tmp1
        tmp71 = tmp69 < 0
        tmp72 = tl.where(tmp71, tmp70, tmp69)
        tmp73 = tl.load(in_ptr2 + (tmp72 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp74 = tl.load(in_ptr3 + (tmp72 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp75 = tl.load(in_ptr4 + (tmp72 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.where(tmp18, tmp74, tmp75)
        tmp77 = tl.where(tmp13, tmp73, tmp76)
        tmp78 = tl.load(in_ptr5 + (tmp72 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tmp77 + tmp78
        tmp80 = tmp36 - tmp37
        tmp81 = tmp80 * tmp61
        tmp82 = tmp81 - tmp63
        tmp83 = tmp82 * tmp80
        tmp84 = tmp83 * tmp80
        tmp85 = tmp84 + tmp36
        tmp86 = tmp79 * tmp85
        tmp88 = tmp87 + tmp1
        tmp89 = tmp87 < 0
        tmp90 = tl.where(tmp89, tmp88, tmp87)
        tmp91 = tl.load(in_ptr2 + (tmp90 + 8*tmp4 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp92 = tl.load(in_ptr3 + (tmp90 + 8*tmp4 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = tl.load(in_ptr4 + (tmp90 + 8*tmp4 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tl.where(tmp18, tmp92, tmp93)
        tmp95 = tl.where(tmp13, tmp91, tmp94)
        tmp96 = tl.load(in_ptr5 + (tmp90 + 8*tmp4 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp97 = tmp95 + tmp96
        tmp98 = 2.0
        tmp99 = tmp98 - tmp37
        tmp100 = tmp99 * tmp39
        tmp101 = tmp100 - tmp41
        tmp102 = tmp101 * tmp99
        tmp103 = tmp102 + tmp44
        tmp104 = tmp103 * tmp99
        tmp105 = tmp104 - tmp47
        tmp106 = tmp97 * tmp105
        tmp108 = tmp107 + tmp1
        tmp109 = tmp107 < 0
        tmp110 = tl.where(tmp109, tmp108, tmp107)
        tmp111 = tl.load(in_ptr2 + (tmp8 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp112 = tl.load(in_ptr3 + (tmp8 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp113 = tl.load(in_ptr4 + (tmp8 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp114 = tl.where(tmp18, tmp112, tmp113)
        tmp115 = tl.where(tmp13, tmp111, tmp114)
        tmp116 = tl.load(in_ptr5 + (tmp8 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp117 = tmp115 + tmp116
        tmp118 = tmp117 * tmp48
        tmp119 = tl.load(in_ptr2 + (tmp53 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp120 = tl.load(in_ptr3 + (tmp53 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp121 = tl.load(in_ptr4 + (tmp53 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp122 = tl.where(tmp18, tmp120, tmp121)
        tmp123 = tl.where(tmp13, tmp119, tmp122)
        tmp124 = tl.load(in_ptr5 + (tmp53 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp125 = tmp123 + tmp124
        tmp126 = tmp125 * tmp67
        tmp127 = tl.load(in_ptr2 + (tmp72 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp128 = tl.load(in_ptr3 + (tmp72 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp129 = tl.load(in_ptr4 + (tmp72 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp130 = tl.where(tmp18, tmp128, tmp129)
        tmp131 = tl.where(tmp13, tmp127, tmp130)
        tmp132 = tl.load(in_ptr5 + (tmp72 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp133 = tmp131 + tmp132
        tmp134 = tmp133 * tmp85
        tmp136 = tmp135 + tmp1
        tmp137 = tmp135 < 0
        tmp138 = tl.where(tmp137, tmp136, tmp135)
        tmp139 = tl.load(in_ptr2 + (tmp8 + 8*tmp138 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp140 = tl.load(in_ptr3 + (tmp8 + 8*tmp138 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp141 = tl.load(in_ptr4 + (tmp8 + 8*tmp138 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp142 = tl.where(tmp18, tmp140, tmp141)
        tmp143 = tl.where(tmp13, tmp139, tmp142)
        tmp144 = tl.load(in_ptr5 + (tmp8 + 8*tmp138 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp145 = tmp143 + tmp144
        tmp146 = tmp145 * tmp48
        tmp147 = tl.load(in_ptr2 + (tmp90 + 8*tmp110 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp148 = tl.load(in_ptr3 + (tmp90 + 8*tmp110 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp149 = tl.load(in_ptr4 + (tmp90 + 8*tmp110 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.where(tmp18, tmp148, tmp149)
        tmp151 = tl.where(tmp13, tmp147, tmp150)
        tmp152 = tl.load(in_ptr5 + (tmp90 + 8*tmp110 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp153 = tmp151 + tmp152
        tmp154 = tmp153 * tmp105
        tmp155 = tl.load(in_ptr2 + (tmp53 + 8*tmp138 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp156 = tl.load(in_ptr3 + (tmp53 + 8*tmp138 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp157 = tl.load(in_ptr4 + (tmp53 + 8*tmp138 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp158 = tl.where(tmp18, tmp156, tmp157)
        tmp159 = tl.where(tmp13, tmp155, tmp158)
        tmp160 = tl.load(in_ptr5 + (tmp53 + 8*tmp138 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp161 = tmp159 + tmp160
        tmp162 = tmp161 * tmp67
        tmp163 = tl.load(in_ptr2 + (tmp72 + 8*tmp138 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp164 = tl.load(in_ptr3 + (tmp72 + 8*tmp138 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp165 = tl.load(in_ptr4 + (tmp72 + 8*tmp138 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp166 = tl.where(tmp18, tmp164, tmp165)
        tmp167 = tl.where(tmp13, tmp163, tmp166)
        tmp168 = tl.load(in_ptr5 + (tmp72 + 8*tmp138 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp169 = tmp167 + tmp168
        tmp170 = tmp169 * tmp85
        tmp171 = tl.load(in_ptr2 + (tmp90 + 8*tmp138 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp172 = tl.load(in_ptr3 + (tmp90 + 8*tmp138 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp173 = tl.load(in_ptr4 + (tmp90 + 8*tmp138 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp174 = tl.where(tmp18, tmp172, tmp173)
        tmp175 = tl.where(tmp13, tmp171, tmp174)
        tmp176 = tl.load(in_ptr5 + (tmp90 + 8*tmp138 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp177 = tmp175 + tmp176
        tmp178 = tmp177 * tmp105
        tmp180 = tmp179 + tmp1
        tmp181 = tmp179 < 0
        tmp182 = tl.where(tmp181, tmp180, tmp179)
        tmp183 = tl.load(in_ptr2 + (tmp8 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp184 = tl.load(in_ptr3 + (tmp8 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp185 = tl.load(in_ptr4 + (tmp8 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp186 = tl.where(tmp18, tmp184, tmp185)
        tmp187 = tl.where(tmp13, tmp183, tmp186)
        tmp188 = tl.load(in_ptr5 + (tmp8 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp189 = tmp187 + tmp188
        tmp190 = tmp189 * tmp48
        tmp191 = tl.load(in_ptr2 + (tmp53 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp192 = tl.load(in_ptr3 + (tmp53 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp193 = tl.load(in_ptr4 + (tmp53 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp194 = tl.where(tmp18, tmp192, tmp193)
        tmp195 = tl.where(tmp13, tmp191, tmp194)
        tmp196 = tl.load(in_ptr5 + (tmp53 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp197 = tmp195 + tmp196
        tmp198 = tmp197 * tmp67
        tmp199 = tl.load(in_ptr2 + (tmp72 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp200 = tl.load(in_ptr3 + (tmp72 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp201 = tl.load(in_ptr4 + (tmp72 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp202 = tl.where(tmp18, tmp200, tmp201)
        tmp203 = tl.where(tmp13, tmp199, tmp202)
        tmp204 = tl.load(in_ptr5 + (tmp72 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp205 = tmp203 + tmp204
        tmp206 = tmp205 * tmp85
        tmp207 = tl.load(in_ptr2 + (tmp90 + 8*tmp182 + 64*(r4 + 8*x0) + 8192*x1), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp208 = tl.load(in_ptr3 + (tmp90 + 8*tmp182 + 64*((-128) + r4 + 8*x0) + 4096*x1), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp209 = tl.load(in_ptr4 + (tmp90 + 8*tmp182 + 64*((-192) + r4 + 8*x0) + 4096*x1), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp210 = tl.where(tmp18, tmp208, tmp209)
        tmp211 = tl.where(tmp13, tmp207, tmp210)
        tmp212 = tl.load(in_ptr5 + (tmp90 + 8*tmp182 + 64*r4 + 512*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp213 = tmp211 + tmp212
        tmp214 = tmp213 * tmp105
        tmp215 = tmp49 + tmp68
        tmp216 = tmp215 + tmp86
        tmp217 = tmp216 + tmp106
        tmp218 = r3
        tmp219 = tmp218.to(tl.float32)
        tmp220 = tmp219 * tmp30
        tmp221 = libdevice.floor(tmp220)
        tmp222 = tmp220 - tmp221
        tmp223 = triton_helpers.maximum(tmp222, tmp34)
        tmp224 = triton_helpers.minimum(tmp223, tmp36)
        tmp225 = tmp224 + tmp36
        tmp226 = tmp225 * tmp39
        tmp227 = tmp226 - tmp41
        tmp228 = tmp227 * tmp225
        tmp229 = tmp228 + tmp44
        tmp230 = tmp229 * tmp225
        tmp231 = tmp230 - tmp47
        tmp232 = tmp217 * tmp231
        tmp233 = tmp118 + tmp126
        tmp234 = tmp233 + tmp134
        tmp235 = tmp234 + tmp154
        tmp236 = tmp224 * tmp61
        tmp237 = tmp236 - tmp63
        tmp238 = tmp237 * tmp224
        tmp239 = tmp238 * tmp224
        tmp240 = tmp239 + tmp36
        tmp241 = tmp235 * tmp240
        tmp242 = tmp232 + tmp241
        tmp243 = tmp146 + tmp162
        tmp244 = tmp243 + tmp170
        tmp245 = tmp244 + tmp178
        tmp246 = tmp36 - tmp224
        tmp247 = tmp246 * tmp61
        tmp248 = tmp247 - tmp63
        tmp249 = tmp248 * tmp246
        tmp250 = tmp249 * tmp246
        tmp251 = tmp250 + tmp36
        tmp252 = tmp245 * tmp251
        tmp253 = tmp242 + tmp252
        tmp254 = tmp190 + tmp198
        tmp255 = tmp254 + tmp206
        tmp256 = tmp255 + tmp214
        tmp257 = tmp98 - tmp224
        tmp258 = tmp257 * tmp39
        tmp259 = tmp258 - tmp41
        tmp260 = tmp259 * tmp257
        tmp261 = tmp260 + tmp44
        tmp262 = tmp261 * tmp257
        tmp263 = tmp262 - tmp47
        tmp264 = tmp256 * tmp263
        tmp265 = tmp253 + tmp264
        tmp266 = tl.load(in_ptr12 + (r6 + 256*(r4 + 8*x0) + 32768*x1), rmask & tmp13 & xmask, eviction_policy='evict_first', other=0.0)
        tmp267 = tl.load(in_ptr13 + (r6 + 256*((-128) + r4 + 8*x0) + 16384*x1), rmask & tmp18 & xmask, eviction_policy='evict_first', other=0.0)
        tmp268 = tl.load(in_ptr14 + (r6 + 256*((-192) + r4 + 8*x0) + 16384*x1), rmask & tmp20 & xmask, eviction_policy='evict_first', other=0.0)
        tmp269 = tl.where(tmp18, tmp267, tmp268)
        tmp270 = tl.where(tmp13, tmp266, tmp269)
        tmp272 = tmp270 + tmp271
        tmp273 = tmp272 + tmp265
        tmp274 = tl.broadcast_to(tmp273, [XBLOCK, RBLOCK])
        tmp275_mean_next, tmp275_m2_next, tmp275_weight_next = triton_helpers.welford_reduce(
            tmp274, tmp275_mean, tmp275_m2, tmp275_weight, roffset == 0
        )
        tmp275_mean = tl.where(rmask & xmask, tmp275_mean_next, tmp275_mean)
        tmp275_m2 = tl.where(rmask & xmask, tmp275_m2_next, tmp275_m2)
        tmp275_weight = tl.where(rmask & xmask, tmp275_weight_next, tmp275_weight)
        tl.store(in_out_ptr0 + (r7 + 2048*x5), tmp273, rmask & xmask)
    tmp275_tmp, tmp276_tmp, tmp277_tmp = triton_helpers.welford(
        tmp275_mean, tmp275_m2, tmp275_weight, 1
    )
    tmp275 = tmp275_tmp[:, None]
    tmp276 = tmp276_tmp[:, None]
    tmp277 = tmp277_tmp[:, None]
    tl.store(out_ptr15 + (x5), tmp275, xmask)
    tl.store(out_ptr16 + (x5), tmp276, xmask)
    tmp278 = 2048.0
    tmp279 = tmp276 / tmp278
    tmp280 = 1e-05
    tmp281 = tmp279 + tmp280
    tmp282 = libdevice.rsqrt(tmp281)
    tl.store(out_ptr17 + (x5), tmp282, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, ), (1, ))
    assert_size_stride(primals_83, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_108, (64, ), (1, ))
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_111, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, ), (1, ))
    assert_size_stride(primals_147, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_157, (256, ), (1, ))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (64, ), (1, ))
    assert_size_stride(primals_165, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_168, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (64, ), (1, ))
    assert_size_stride(primals_183, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_186, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_193, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (256, ), (1, ))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_218, (64, ), (1, ))
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, ), (1, ))
    assert_size_stride(primals_223, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_227, (64, ), (1, ))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_242, (128, ), (1, ))
    assert_size_stride(primals_243, (128, ), (1, ))
    assert_size_stride(primals_244, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_257, (256, ), (1, ))
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, ), (1, ))
    assert_size_stride(primals_265, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_266, (256, ), (1, ))
    assert_size_stride(primals_267, (256, ), (1, ))
    assert_size_stride(primals_268, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_275, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (256, ), (1, ))
    assert_size_stride(primals_279, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_280, (256, ), (1, ))
    assert_size_stride(primals_281, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_288, (128, ), (1, ))
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_294, (256, ), (1, ))
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_304, (256, ), (1, ))
    assert_size_stride(primals_305, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_306, (128, ), (1, ))
    assert_size_stride(primals_307, (128, ), (1, ))
    assert_size_stride(primals_308, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_312, (256, ), (1, ))
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_321, (256, ), (1, ))
    assert_size_stride(primals_322, (256, ), (1, ))
    assert_size_stride(primals_323, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_326, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_327, (64, ), (1, ))
    assert_size_stride(primals_328, (64, ), (1, ))
    assert_size_stride(primals_329, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_330, (256, ), (1, ))
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_339, (256, ), (1, ))
    assert_size_stride(primals_340, (256, ), (1, ))
    assert_size_stride(primals_341, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, ), (1, ))
    assert_size_stride(primals_347, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_348, (256, ), (1, ))
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_351, (128, ), (1, ))
    assert_size_stride(primals_352, (128, ), (1, ))
    assert_size_stride(primals_353, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_354, (64, ), (1, ))
    assert_size_stride(primals_355, (64, ), (1, ))
    assert_size_stride(primals_356, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_357, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_358, (256, ), (1, ))
    assert_size_stride(primals_359, (256, ), (1, ))
    assert_size_stride(primals_360, (256, ), (1, ))
    assert_size_stride(primals_361, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_362, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf3 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf5 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d, group_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf1, primals_2, buf2, buf3, buf5, 128, 2048, grid=grid(128), stream=stream0)
        del primals_2
        buf6 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf8 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf25 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf26 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf10 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf28 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm, x, out1, input_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_relu_1.run(buf1, buf2, buf3, primals_4, primals_5, buf6, buf7, buf8, buf25, buf26, buf10, buf28, 128, 2048, grid=grid(128), stream=stream0)
        del primals_5
        buf11 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf29 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1, out1_1, input_1, input_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_2.run(buf6, buf7, buf8, primals_6, primals_7, buf25, buf26, primals_15, primals_16, buf11, buf29, 262144, grid=grid(262144), stream=stream0)
        del primals_16
        del primals_7
        # Topologically Sorted Source Nodes: [out1_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf13 = buf8; del buf8  # reuse
        buf14 = buf26; del buf26  # reuse
        buf16 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_3.run(buf12, buf13, buf14, buf16, 128, 2048, grid=grid(128), stream=stream0)
        buf17 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2, out2_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_4.run(buf12, buf13, buf14, primals_9, primals_10, buf17, 262144, grid=grid(262144), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [out2_2], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf19 = buf14; del buf14  # reuse
        buf23 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3, out3_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_relu_5.run(buf18, primals_12, primals_13, buf19, buf23, buf22, 128, 1024, grid=grid(128), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [out3_2], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 32, 32), (32768, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [out3_3, out3_4], Original ATen: [aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_6.run(buf31, buf12, buf18, buf24, 524288, grid=grid(524288), stream=stream0)
        buf32 = reinterpret_tensor(buf24, (4, 128, 16, 16), (32768, 256, 16, 1), 0); del buf24  # reuse
        buf33 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf34 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf36 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, out1_3], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_7.run(buf31, buf32, buf33, buf34, buf36, 128, 1024, grid=grid(128), stream=stream0)
        buf37 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_3, out1_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf32, buf33, buf34, primals_18, primals_19, buf37, 131072, grid=grid(131072), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [out1_5], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf39 = buf34; del buf34  # reuse
        buf40 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf42 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf38, buf39, buf40, buf42, 128, 512, grid=grid(128), stream=stream0)
        buf43 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_3, out2_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf38, buf39, buf40, primals_21, primals_22, buf43, 65536, grid=grid(65536), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [out2_5], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf45 = buf40; del buf40  # reuse
        buf49 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf48 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_5, out3_6], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_relu_11.run(buf44, primals_24, primals_25, buf45, buf49, buf48, 128, 256, grid=grid(128), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [out3_7], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf51 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf52 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf53 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf55 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_8, out1_6], Original ATen: [aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_group_norm_12.run(buf38, buf44, buf50, buf32, buf51, buf52, buf53, buf55, 128, 1024, grid=grid(128), stream=stream0)
        buf56 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf70 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_6, out1_7, input_4, input_5], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_13.run(buf51, buf32, buf52, buf53, primals_27, primals_28, primals_36, primals_37, buf56, buf70, 131072, grid=grid(131072), stream=stream0)
        del primals_28
        del primals_37
        # Topologically Sorted Source Nodes: [out1_8], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf58 = buf53; del buf53  # reuse
        buf59 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf61 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf57, buf58, buf59, buf61, 128, 1024, grid=grid(128), stream=stream0)
        buf62 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_6, out2_7], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf57, buf58, buf59, primals_30, primals_31, buf62, 131072, grid=grid(131072), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [out2_8], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf64 = buf59; del buf59  # reuse
        buf65 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf67 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf63, buf64, buf65, buf67, 128, 512, grid=grid(128), stream=stream0)
        buf68 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_10, out3_11], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf63, buf64, buf65, primals_33, primals_34, buf68, 65536, grid=grid(65536), stream=stream0)
        del primals_34
        # Topologically Sorted Source Nodes: [out3_12], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf72 = buf71; del buf71  # reuse
        buf73 = buf65; del buf65  # reuse
        buf74 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf76 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_13, out3_14, out1_9], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_group_norm_15.run(buf72, buf57, buf63, buf69, buf73, buf74, buf76, 128, 2048, grid=grid(128), stream=stream0)
        buf77 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_9, out1_10], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf72, buf73, buf74, primals_39, primals_40, buf77, 262144, grid=grid(262144), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [out1_11], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf79 = buf74; del buf74  # reuse
        buf80 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf82 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf78, buf79, buf80, buf82, 128, 1024, grid=grid(128), stream=stream0)
        buf83 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_9, out2_10], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf78, buf79, buf80, primals_42, primals_43, buf83, 131072, grid=grid(131072), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [out2_11], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf85 = buf80; del buf80  # reuse
        buf86 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf88 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_15], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf84, buf85, buf86, buf88, 128, 512, grid=grid(128), stream=stream0)
        buf89 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [out3_15, out3_16], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf84, buf85, buf86, primals_45, primals_46, buf89, 65536, grid=grid(65536), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [out3_17], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf91 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf92 = buf86; del buf86  # reuse
        buf93 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf95 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1, out1_12], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_17.run(buf72, buf91, buf92, buf93, buf95, 128, 512, grid=grid(128), stream=stream0)
        buf96 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_12, out1_13], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf91, buf92, buf93, primals_48, primals_49, buf96, 65536, grid=grid(65536), stream=stream0)
        del primals_49
        # Topologically Sorted Source Nodes: [out1_14], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf98 = buf93; del buf93  # reuse
        buf99 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf101 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_12], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf97, buf98, buf99, buf101, 128, 256, grid=grid(128), stream=stream0)
        buf102 = reinterpret_tensor(buf50, (4, 128, 8, 8), (8192, 64, 8, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [out2_12, out2_13], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf97, buf98, buf99, primals_51, primals_52, buf102, 32768, grid=grid(32768), stream=stream0)
        del primals_52
        # Topologically Sorted Source Nodes: [out2_14], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf104 = buf99; del buf99  # reuse
        buf105 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf107 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf103, buf104, buf105, buf107, 128, 128, grid=grid(128), stream=stream0)
        buf108 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_20, out3_21], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf103, buf104, buf105, primals_54, primals_55, buf108, 16384, grid=grid(16384), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [out3_22], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf110 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf111 = buf105; del buf105  # reuse
        buf112 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf114 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_23, out3_24, out1_15], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_23.run(buf97, buf103, buf109, buf91, buf110, buf111, buf112, buf114, 128, 512, grid=grid(128), stream=stream0)
        buf115 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_15, out1_16], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf110, buf111, buf112, primals_57, primals_58, buf115, 65536, grid=grid(65536), stream=stream0)
        del primals_58
        # Topologically Sorted Source Nodes: [out1_17], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf117 = buf112; del buf112  # reuse
        buf118 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf120 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_15], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf116, buf117, buf118, buf120, 128, 256, grid=grid(128), stream=stream0)
        buf121 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_15, out2_16], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf116, buf117, buf118, primals_60, primals_61, buf121, 32768, grid=grid(32768), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [out2_17], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf123 = buf118; del buf118  # reuse
        buf124 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf126 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_25], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf122, buf123, buf124, buf126, 128, 128, grid=grid(128), stream=stream0)
        buf127 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [out3_25, out3_26], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf122, buf123, buf124, primals_63, primals_64, buf127, 16384, grid=grid(16384), stream=stream0)
        del primals_64
        # Topologically Sorted Source Nodes: [out3_27], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf129 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf130 = buf124; del buf124  # reuse
        buf131 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf133 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_1, out1_18], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_24.run(buf110, buf129, buf130, buf131, buf133, 128, 128, grid=grid(128), stream=stream0)
        buf134 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_18, out1_19], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf129, buf130, buf131, primals_66, primals_67, buf134, 16384, grid=grid(16384), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [out1_20], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf136 = buf131; del buf131  # reuse
        buf137 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf139 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf135, buf136, buf137, buf139, 128, 64, grid=grid(128), stream=stream0)
        buf140 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_18, out2_19], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf135, buf136, buf137, primals_69, primals_70, buf140, 8192, grid=grid(8192), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [out2_20], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf142 = buf137; del buf137  # reuse
        buf143 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf145 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_30], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf141, buf142, buf143, buf145, 128, 32, grid=grid(128), stream=stream0)
        buf146 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_30, out3_31], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf141, buf142, buf143, primals_72, primals_73, buf146, 4096, grid=grid(4096), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [out3_32], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf148 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf149 = buf143; del buf143  # reuse
        buf150 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf152 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_33, out1_21], Original ATen: [aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_group_norm_30.run(buf135, buf141, buf147, buf129, buf148, buf149, buf150, buf152, 128, 128, grid=grid(128), stream=stream0)
        buf153 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_21, out1_22], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_31.run(buf148, buf129, buf149, buf150, primals_75, primals_76, buf153, 16384, grid=grid(16384), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [out1_23], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf155 = buf150; del buf150  # reuse
        buf156 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf158 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_21], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf154, buf155, buf156, buf158, 128, 64, grid=grid(128), stream=stream0)
        buf159 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_21, out2_22], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf154, buf155, buf156, primals_78, primals_79, buf159, 8192, grid=grid(8192), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [out2_23], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf161 = buf156; del buf156  # reuse
        buf162 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf164 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf160, buf161, buf162, buf164, 128, 32, grid=grid(128), stream=stream0)
        buf165 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [out3_35, out3_36], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf160, buf161, buf162, primals_81, primals_82, buf165, 4096, grid=grid(4096), stream=stream0)
        del primals_82
        # Topologically Sorted Source Nodes: [out3_37], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf167 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf168 = buf162; del buf162  # reuse
        buf169 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf171 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_34, out3_38, out3_39, out1_24], Original ATen: [aten.add, aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_32.run(buf154, buf160, buf166, buf148, buf129, buf167, buf168, buf169, buf171, 128, 128, grid=grid(128), stream=stream0)
        buf172 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_24, out1_25], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf167, buf168, buf169, primals_84, primals_85, buf172, 16384, grid=grid(16384), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [out1_26], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf174 = buf169; del buf169  # reuse
        buf175 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf177 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf173, buf174, buf175, buf177, 128, 64, grid=grid(128), stream=stream0)
        buf178 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_24, out2_25], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf173, buf174, buf175, primals_87, primals_88, buf178, 8192, grid=grid(8192), stream=stream0)
        del primals_88
        # Topologically Sorted Source Nodes: [out2_26], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf180 = buf175; del buf175  # reuse
        buf181 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf183 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_40], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf179, buf180, buf181, buf183, 128, 32, grid=grid(128), stream=stream0)
        buf184 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [out3_40, out3_41], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf179, buf180, buf181, primals_90, primals_91, buf184, 4096, grid=grid(4096), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [out3_42], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf186 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_33.run(buf186, 8, grid=grid(8), stream=stream0)
        buf187 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_33.run(buf187, 8, grid=grid(8), stream=stream0)
        buf188 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_34.run(buf188, 8, grid=grid(8), stream=stream0)
        buf189 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35.run(buf189, 8, grid=grid(8), stream=stream0)
        buf190 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36.run(buf190, 8, grid=grid(8), stream=stream0)
        buf195 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_34.run(buf195, 8, grid=grid(8), stream=stream0)
        buf200 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_35.run(buf200, 8, grid=grid(8), stream=stream0)
        buf205 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_36.run(buf205, 8, grid=grid(8), stream=stream0)
        buf191 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf210 = buf191; del buf191  # reuse
        buf211 = buf210; del buf210  # reuse
        buf212 = buf211; del buf211  # reuse
        buf213 = buf181; del buf181  # reuse
        buf214 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf216 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_28, out3_29, out3_43, out3_44, up2, low2, out1_27], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_37.run(buf212, buf186, buf187, buf173, buf179, buf185, buf167, buf188, buf189, buf190, buf195, buf200, buf205, buf116, buf122, buf128, buf110, buf213, buf214, buf216, 128, 512, grid=grid(128), stream=stream0)
        buf217 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_27, out1_28], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf212, buf213, buf214, primals_93, primals_94, buf217, 65536, grid=grid(65536), stream=stream0)
        del primals_94
        # Topologically Sorted Source Nodes: [out1_29], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf219 = buf214; del buf214  # reuse
        buf220 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf222 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_27], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf218, buf219, buf220, buf222, 128, 256, grid=grid(128), stream=stream0)
        buf223 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_27, out2_28], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf218, buf219, buf220, primals_96, primals_97, buf223, 32768, grid=grid(32768), stream=stream0)
        del primals_97
        # Topologically Sorted Source Nodes: [out2_29], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf225 = buf220; del buf220  # reuse
        buf226 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf228 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_45], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf224, buf225, buf226, buf228, 128, 128, grid=grid(128), stream=stream0)
        buf229 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [out3_45, out3_46], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf224, buf225, buf226, primals_99, primals_100, buf229, 16384, grid=grid(16384), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [out3_47], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf231 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_38.run(buf231, 16, grid=grid(16), stream=stream0)
        buf232 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_38.run(buf232, 16, grid=grid(16), stream=stream0)
        buf233 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_39.run(buf233, 16, grid=grid(16), stream=stream0)
        buf234 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40.run(buf234, 16, grid=grid(16), stream=stream0)
        buf235 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41.run(buf235, 16, grid=grid(16), stream=stream0)
        buf240 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.floor, aten._to_copy, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_39.run(buf240, 16, grid=grid(16), stream=stream0)
        buf245 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_40.run(buf245, 16, grid=grid(16), stream=stream0)
        buf250 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_41.run(buf250, 16, grid=grid(16), stream=stream0)
        buf236 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf255 = buf236; del buf236  # reuse
        buf256 = buf255; del buf255  # reuse
        buf257 = buf256; del buf256  # reuse
        buf258 = buf226; del buf226  # reuse
        buf259 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf261 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_18, out3_19, out3_48, out3_49, up2_1, hg, out1_30], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_42.run(buf257, buf231, buf232, buf218, buf224, buf230, buf212, buf233, buf234, buf235, buf240, buf245, buf250, buf78, buf84, buf90, buf72, buf258, buf259, buf261, 128, 2048, grid=grid(128), stream=stream0)
        buf262 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_30, out1_31], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf257, buf258, buf259, primals_102, primals_103, buf262, 262144, grid=grid(262144), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [out1_32], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf264 = buf259; del buf259  # reuse
        buf265 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf267 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_30], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf263, buf264, buf265, buf267, 128, 1024, grid=grid(128), stream=stream0)
        buf268 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_30, out2_31], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf263, buf264, buf265, primals_105, primals_106, buf268, 131072, grid=grid(131072), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [out2_32], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf270 = buf265; del buf265  # reuse
        buf271 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf273 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_50], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf269, buf270, buf271, buf273, 128, 512, grid=grid(128), stream=stream0)
        buf274 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [out3_50, out3_51], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf269, buf270, buf271, primals_108, primals_109, buf274, 65536, grid=grid(65536), stream=stream0)
        del primals_109
        # Topologically Sorted Source Nodes: [out3_52], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf276 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_53, out3_54], Original ATen: [aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_43.run(buf263, buf269, buf275, buf257, buf276, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf278 = buf277; del buf277  # reuse
        buf279 = buf271; del buf271  # reuse
        buf280 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf282 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_36, group_norm_36], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_44.run(buf278, primals_112, buf279, buf280, buf282, 128, 2048, grid=grid(128), stream=stream0)
        del primals_112
        buf283 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_36, ll], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf278, buf279, buf280, primals_113, primals_114, buf283, 262144, grid=grid(262144), stream=stream0)
        del primals_114
        # Topologically Sorted Source Nodes: [tmp_out], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf285 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [tmp_out], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_45.run(buf285, primals_116, 262144, grid=grid(262144), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [ll_1], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf283, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [tmp_out_], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf285, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf288 = buf286; del buf286  # reuse
        buf289 = buf280; del buf280  # reuse
        buf290 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf292 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [ll_1, tmp_out_, add_2, previous, out1_33], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_46.run(buf288, buf72, primals_118, buf287, primals_120, buf289, buf290, buf292, 128, 2048, grid=grid(128), stream=stream0)
        del primals_118
        del primals_120
        buf293 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [out1_33, out1_34], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf288, buf289, buf290, primals_121, primals_122, buf293, 262144, grid=grid(262144), stream=stream0)
        del primals_122
        # Topologically Sorted Source Nodes: [out1_35], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf295 = buf290; del buf290  # reuse
        buf296 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf298 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_33], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf294, buf295, buf296, buf298, 128, 1024, grid=grid(128), stream=stream0)
        buf299 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_33, out2_34], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf294, buf295, buf296, primals_124, primals_125, buf299, 131072, grid=grid(131072), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [out2_35], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf301 = buf296; del buf296  # reuse
        buf302 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf304 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_55], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf300, buf301, buf302, buf304, 128, 512, grid=grid(128), stream=stream0)
        buf305 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [out3_55, out3_56], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf300, buf301, buf302, primals_127, primals_128, buf305, 65536, grid=grid(65536), stream=stream0)
        del primals_128
        # Topologically Sorted Source Nodes: [out3_57], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf307 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf308 = buf302; del buf302  # reuse
        buf309 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf311 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_2, out1_36], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_17.run(buf288, buf307, buf308, buf309, buf311, 128, 512, grid=grid(128), stream=stream0)
        buf312 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_36, out1_37], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf307, buf308, buf309, primals_130, primals_131, buf312, 65536, grid=grid(65536), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [out1_38], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf314 = buf309; del buf309  # reuse
        buf315 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf317 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_36], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf313, buf314, buf315, buf317, 128, 256, grid=grid(128), stream=stream0)
        buf318 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_36, out2_37], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf313, buf314, buf315, primals_133, primals_134, buf318, 32768, grid=grid(32768), stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [out2_38], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf320 = buf315; del buf315  # reuse
        buf321 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf323 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_60], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf319, buf320, buf321, buf323, 128, 128, grid=grid(128), stream=stream0)
        buf324 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [out3_60, out3_61], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf319, buf320, buf321, primals_136, primals_137, buf324, 16384, grid=grid(16384), stream=stream0)
        del primals_137
        # Topologically Sorted Source Nodes: [out3_62], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf326 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf327 = buf321; del buf321  # reuse
        buf328 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf330 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_63, out3_64, out1_39], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_23.run(buf313, buf319, buf325, buf307, buf326, buf327, buf328, buf330, 128, 512, grid=grid(128), stream=stream0)
        buf331 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_39, out1_40], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf326, buf327, buf328, primals_139, primals_140, buf331, 65536, grid=grid(65536), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [out1_41], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf333 = buf328; del buf328  # reuse
        buf334 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf336 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf332, buf333, buf334, buf336, 128, 256, grid=grid(128), stream=stream0)
        buf337 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_39, out2_40], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf332, buf333, buf334, primals_142, primals_143, buf337, 32768, grid=grid(32768), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [out2_41], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf339 = buf334; del buf334  # reuse
        buf340 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf342 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_65], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf338, buf339, buf340, buf342, 128, 128, grid=grid(128), stream=stream0)
        buf343 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [out3_65, out3_66], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf338, buf339, buf340, primals_145, primals_146, buf343, 16384, grid=grid(16384), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out3_67], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf345 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf346 = buf340; del buf340  # reuse
        buf347 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf349 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_3, out1_42], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_24.run(buf326, buf345, buf346, buf347, buf349, 128, 128, grid=grid(128), stream=stream0)
        buf350 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_42, out1_43], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf345, buf346, buf347, primals_148, primals_149, buf350, 16384, grid=grid(16384), stream=stream0)
        del primals_149
        # Topologically Sorted Source Nodes: [out1_44], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf352 = buf347; del buf347  # reuse
        buf353 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf355 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_42], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf351, buf352, buf353, buf355, 128, 64, grid=grid(128), stream=stream0)
        buf356 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_42, out2_43], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf351, buf352, buf353, primals_151, primals_152, buf356, 8192, grid=grid(8192), stream=stream0)
        del primals_152
        # Topologically Sorted Source Nodes: [out2_44], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf356, primals_153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf358 = buf353; del buf353  # reuse
        buf359 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf361 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_70], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf357, buf358, buf359, buf361, 128, 32, grid=grid(128), stream=stream0)
        buf362 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out3_70, out3_71], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf357, buf358, buf359, primals_154, primals_155, buf362, 4096, grid=grid(4096), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [out3_72], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf364 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf365 = buf359; del buf359  # reuse
        buf366 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf368 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_73, out1_45], Original ATen: [aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_group_norm_30.run(buf351, buf357, buf363, buf345, buf364, buf365, buf366, buf368, 128, 128, grid=grid(128), stream=stream0)
        buf369 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_45, out1_46], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_31.run(buf364, buf345, buf365, buf366, primals_157, primals_158, buf369, 16384, grid=grid(16384), stream=stream0)
        del primals_158
        # Topologically Sorted Source Nodes: [out1_47], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf371 = buf366; del buf366  # reuse
        buf372 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf374 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_45], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf370, buf371, buf372, buf374, 128, 64, grid=grid(128), stream=stream0)
        buf375 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_45, out2_46], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf370, buf371, buf372, primals_160, primals_161, buf375, 8192, grid=grid(8192), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [out2_47], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf377 = buf372; del buf372  # reuse
        buf378 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf380 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_75], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf376, buf377, buf378, buf380, 128, 32, grid=grid(128), stream=stream0)
        buf381 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [out3_75, out3_76], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf376, buf377, buf378, primals_163, primals_164, buf381, 4096, grid=grid(4096), stream=stream0)
        del primals_164
        # Topologically Sorted Source Nodes: [out3_77], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf383 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf384 = buf378; del buf378  # reuse
        buf385 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf387 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_74, out3_78, out3_79, out1_48], Original ATen: [aten.add, aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_32.run(buf370, buf376, buf382, buf364, buf345, buf383, buf384, buf385, buf387, 128, 128, grid=grid(128), stream=stream0)
        buf388 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_48, out1_49], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf383, buf384, buf385, primals_166, primals_167, buf388, 16384, grid=grid(16384), stream=stream0)
        del primals_167
        # Topologically Sorted Source Nodes: [out1_50], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf390 = buf385; del buf385  # reuse
        buf391 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf393 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_48], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf389, buf390, buf391, buf393, 128, 64, grid=grid(128), stream=stream0)
        buf394 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_48, out2_49], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf389, buf390, buf391, primals_169, primals_170, buf394, 8192, grid=grid(8192), stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [out2_50], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf396 = buf391; del buf391  # reuse
        buf397 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf399 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_80], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf395, buf396, buf397, buf399, 128, 32, grid=grid(128), stream=stream0)
        buf400 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [out3_80, out3_81], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf395, buf396, buf397, primals_172, primals_173, buf400, 4096, grid=grid(4096), stream=stream0)
        del primals_173
        # Topologically Sorted Source Nodes: [out3_82], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf402 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf418 = buf402; del buf402  # reuse
        buf419 = buf418; del buf418  # reuse
        buf420 = buf419; del buf419  # reuse
        buf421 = buf397; del buf397  # reuse
        buf422 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf424 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2, out3_68, out3_69, out3_83, out3_84, up2_2, low2_1, out1_51], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47.run(buf420, buf186, buf187, buf389, buf395, buf401, buf383, buf188, buf189, buf190, buf195, buf200, buf205, buf332, buf338, buf344, buf326, buf421, buf422, buf424, 128, 512, grid=grid(128), stream=stream0)
        buf425 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_51, out1_52], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf420, buf421, buf422, primals_175, primals_176, buf425, 65536, grid=grid(65536), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [out1_53], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf427 = buf422; del buf422  # reuse
        buf428 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf430 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf426, buf427, buf428, buf430, 128, 256, grid=grid(128), stream=stream0)
        buf431 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_51, out2_52], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf426, buf427, buf428, primals_178, primals_179, buf431, 32768, grid=grid(32768), stream=stream0)
        del primals_179
        # Topologically Sorted Source Nodes: [out2_53], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf433 = buf428; del buf428  # reuse
        buf434 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf436 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_85], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf432, buf433, buf434, buf436, 128, 128, grid=grid(128), stream=stream0)
        buf437 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [out3_85, out3_86], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf432, buf433, buf434, primals_181, primals_182, buf437, 16384, grid=grid(16384), stream=stream0)
        del primals_182
        # Topologically Sorted Source Nodes: [out3_87], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf439 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf455 = buf439; del buf439  # reuse
        buf456 = buf455; del buf455  # reuse
        buf457 = buf456; del buf456  # reuse
        buf458 = buf434; del buf434  # reuse
        buf459 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf461 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2_1, out3_58, out3_59, out3_88, out3_89, up2_3, hg_1, out1_54], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48.run(buf457, buf231, buf232, buf426, buf432, buf438, buf420, buf233, buf234, buf235, buf240, buf245, buf250, buf294, buf300, buf306, buf288, buf458, buf459, buf461, 128, 2048, grid=grid(128), stream=stream0)
        buf462 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_54, out1_55], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf457, buf458, buf459, primals_184, primals_185, buf462, 262144, grid=grid(262144), stream=stream0)
        del primals_185
        # Topologically Sorted Source Nodes: [out1_56], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf464 = buf459; del buf459  # reuse
        buf465 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf467 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_54], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf463, buf464, buf465, buf467, 128, 1024, grid=grid(128), stream=stream0)
        buf468 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_54, out2_55], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf463, buf464, buf465, primals_187, primals_188, buf468, 131072, grid=grid(131072), stream=stream0)
        del primals_188
        # Topologically Sorted Source Nodes: [out2_56], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf468, primals_189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf470 = buf465; del buf465  # reuse
        buf471 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf473 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_90], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf469, buf470, buf471, buf473, 128, 512, grid=grid(128), stream=stream0)
        buf474 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [out3_90, out3_91], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf469, buf470, buf471, primals_190, primals_191, buf474, 65536, grid=grid(65536), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out3_92], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf476 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_93, out3_94], Original ATen: [aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_43.run(buf463, buf469, buf475, buf457, buf476, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf478 = buf477; del buf477  # reuse
        buf479 = buf471; del buf471  # reuse
        buf480 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf482 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_64, group_norm_61], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_44.run(buf478, primals_194, buf479, buf480, buf482, 128, 2048, grid=grid(128), stream=stream0)
        del primals_194
        buf483 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_61, ll_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf478, buf479, buf480, primals_195, primals_196, buf483, 262144, grid=grid(262144), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [tmp_out_1], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf485 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [tmp_out_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_45.run(buf485, primals_198, 262144, grid=grid(262144), stream=stream0)
        del primals_198
        # Topologically Sorted Source Nodes: [ll_3], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf483, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [tmp_out__1], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf485, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf488 = buf486; del buf486  # reuse
        buf489 = buf480; del buf480  # reuse
        buf490 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf492 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [ll_3, tmp_out__1, add_6, previous_1, out1_57], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_46.run(buf488, buf288, primals_200, buf487, primals_202, buf489, buf490, buf492, 128, 2048, grid=grid(128), stream=stream0)
        del primals_200
        del primals_202
        buf493 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [out1_57, out1_58], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf488, buf489, buf490, primals_203, primals_204, buf493, 262144, grid=grid(262144), stream=stream0)
        del primals_204
        # Topologically Sorted Source Nodes: [out1_59], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(buf493, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf495 = buf490; del buf490  # reuse
        buf496 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf498 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_57], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf494, buf495, buf496, buf498, 128, 1024, grid=grid(128), stream=stream0)
        buf499 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_57, out2_58], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf494, buf495, buf496, primals_206, primals_207, buf499, 131072, grid=grid(131072), stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [out2_59], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf501 = buf496; del buf496  # reuse
        buf502 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf504 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_95], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf500, buf501, buf502, buf504, 128, 512, grid=grid(128), stream=stream0)
        buf505 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [out3_95, out3_96], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf500, buf501, buf502, primals_209, primals_210, buf505, 65536, grid=grid(65536), stream=stream0)
        del primals_210
        # Topologically Sorted Source Nodes: [out3_97], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf507 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf508 = buf502; del buf502  # reuse
        buf509 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf511 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_4, out1_60], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_17.run(buf488, buf507, buf508, buf509, buf511, 128, 512, grid=grid(128), stream=stream0)
        buf512 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_60, out1_61], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf507, buf508, buf509, primals_212, primals_213, buf512, 65536, grid=grid(65536), stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [out1_62], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf514 = buf509; del buf509  # reuse
        buf515 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf517 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_60], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf513, buf514, buf515, buf517, 128, 256, grid=grid(128), stream=stream0)
        buf518 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_60, out2_61], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf513, buf514, buf515, primals_215, primals_216, buf518, 32768, grid=grid(32768), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [out2_62], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf520 = buf515; del buf515  # reuse
        buf521 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf523 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_100], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf519, buf520, buf521, buf523, 128, 128, grid=grid(128), stream=stream0)
        buf524 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [out3_100, out3_101], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf519, buf520, buf521, primals_218, primals_219, buf524, 16384, grid=grid(16384), stream=stream0)
        del primals_219
        # Topologically Sorted Source Nodes: [out3_102], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, primals_220, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf526 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf527 = buf521; del buf521  # reuse
        buf528 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf530 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_103, out3_104, out1_63], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_23.run(buf513, buf519, buf525, buf507, buf526, buf527, buf528, buf530, 128, 512, grid=grid(128), stream=stream0)
        buf531 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_63, out1_64], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf526, buf527, buf528, primals_221, primals_222, buf531, 65536, grid=grid(65536), stream=stream0)
        del primals_222
        # Topologically Sorted Source Nodes: [out1_65], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf533 = buf528; del buf528  # reuse
        buf534 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf536 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_63], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf532, buf533, buf534, buf536, 128, 256, grid=grid(128), stream=stream0)
        buf537 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_63, out2_64], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf532, buf533, buf534, primals_224, primals_225, buf537, 32768, grid=grid(32768), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [out2_65], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf539 = buf534; del buf534  # reuse
        buf540 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf542 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_105], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf538, buf539, buf540, buf542, 128, 128, grid=grid(128), stream=stream0)
        buf543 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [out3_105, out3_106], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf538, buf539, buf540, primals_227, primals_228, buf543, 16384, grid=grid(16384), stream=stream0)
        del primals_228
        # Topologically Sorted Source Nodes: [out3_107], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf545 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf546 = buf540; del buf540  # reuse
        buf547 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf549 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_5, out1_66], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_24.run(buf526, buf545, buf546, buf547, buf549, 128, 128, grid=grid(128), stream=stream0)
        buf550 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_66, out1_67], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf545, buf546, buf547, primals_230, primals_231, buf550, 16384, grid=grid(16384), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [out1_68], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf552 = buf547; del buf547  # reuse
        buf553 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf555 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_66], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf551, buf552, buf553, buf555, 128, 64, grid=grid(128), stream=stream0)
        buf556 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_66, out2_67], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf551, buf552, buf553, primals_233, primals_234, buf556, 8192, grid=grid(8192), stream=stream0)
        del primals_234
        # Topologically Sorted Source Nodes: [out2_68], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf558 = buf553; del buf553  # reuse
        buf559 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf561 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_110], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf557, buf558, buf559, buf561, 128, 32, grid=grid(128), stream=stream0)
        buf562 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [out3_110, out3_111], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf557, buf558, buf559, primals_236, primals_237, buf562, 4096, grid=grid(4096), stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [out3_112], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf564 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf565 = buf559; del buf559  # reuse
        buf566 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf568 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_113, out1_69], Original ATen: [aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_group_norm_30.run(buf551, buf557, buf563, buf545, buf564, buf565, buf566, buf568, 128, 128, grid=grid(128), stream=stream0)
        buf569 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_69, out1_70], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_31.run(buf564, buf545, buf565, buf566, primals_239, primals_240, buf569, 16384, grid=grid(16384), stream=stream0)
        del primals_240
        # Topologically Sorted Source Nodes: [out1_71], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf571 = buf566; del buf566  # reuse
        buf572 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf574 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_69], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf570, buf571, buf572, buf574, 128, 64, grid=grid(128), stream=stream0)
        buf575 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_69, out2_70], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf570, buf571, buf572, primals_242, primals_243, buf575, 8192, grid=grid(8192), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [out2_71], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf577 = buf572; del buf572  # reuse
        buf578 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf580 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_115], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf576, buf577, buf578, buf580, 128, 32, grid=grid(128), stream=stream0)
        buf581 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [out3_115, out3_116], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf576, buf577, buf578, primals_245, primals_246, buf581, 4096, grid=grid(4096), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [out3_117], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf583 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf584 = buf578; del buf578  # reuse
        buf585 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf587 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_114, out3_118, out3_119, out1_72], Original ATen: [aten.add, aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_32.run(buf570, buf576, buf582, buf564, buf545, buf583, buf584, buf585, buf587, 128, 128, grid=grid(128), stream=stream0)
        buf588 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_72, out1_73], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf583, buf584, buf585, primals_248, primals_249, buf588, 16384, grid=grid(16384), stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [out1_74], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf590 = buf585; del buf585  # reuse
        buf591 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf593 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_72], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf589, buf590, buf591, buf593, 128, 64, grid=grid(128), stream=stream0)
        buf594 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_72, out2_73], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf589, buf590, buf591, primals_251, primals_252, buf594, 8192, grid=grid(8192), stream=stream0)
        del primals_252
        # Topologically Sorted Source Nodes: [out2_74], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf594, primals_253, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf596 = buf591; del buf591  # reuse
        buf597 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf599 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_120], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf595, buf596, buf597, buf599, 128, 32, grid=grid(128), stream=stream0)
        buf600 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [out3_120, out3_121], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf595, buf596, buf597, primals_254, primals_255, buf600, 4096, grid=grid(4096), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [out3_122], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf602 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf618 = buf602; del buf602  # reuse
        buf619 = buf618; del buf618  # reuse
        buf620 = buf619; del buf619  # reuse
        buf621 = buf597; del buf597  # reuse
        buf622 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf624 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2, out3_108, out3_109, out3_123, out3_124, up2_4, low2_2, out1_75], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47.run(buf620, buf186, buf187, buf589, buf595, buf601, buf583, buf188, buf189, buf190, buf195, buf200, buf205, buf532, buf538, buf544, buf526, buf621, buf622, buf624, 128, 512, grid=grid(128), stream=stream0)
        buf625 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_75, out1_76], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf620, buf621, buf622, primals_257, primals_258, buf625, 65536, grid=grid(65536), stream=stream0)
        del primals_258
        # Topologically Sorted Source Nodes: [out1_77], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf627 = buf622; del buf622  # reuse
        buf628 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf630 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_75], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf626, buf627, buf628, buf630, 128, 256, grid=grid(128), stream=stream0)
        buf631 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_75, out2_76], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf626, buf627, buf628, primals_260, primals_261, buf631, 32768, grid=grid(32768), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [out2_77], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf633 = buf628; del buf628  # reuse
        buf634 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf636 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_125], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf632, buf633, buf634, buf636, 128, 128, grid=grid(128), stream=stream0)
        buf637 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [out3_125, out3_126], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf632, buf633, buf634, primals_263, primals_264, buf637, 16384, grid=grid(16384), stream=stream0)
        del primals_264
        # Topologically Sorted Source Nodes: [out3_127], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, primals_265, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf639 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf655 = buf639; del buf639  # reuse
        buf656 = buf655; del buf655  # reuse
        buf657 = buf656; del buf656  # reuse
        buf658 = buf634; del buf634  # reuse
        buf659 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf661 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2_1, out3_98, out3_99, out3_128, out3_129, up2_5, hg_2, out1_78], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48.run(buf657, buf231, buf232, buf626, buf632, buf638, buf620, buf233, buf234, buf235, buf240, buf245, buf250, buf494, buf500, buf506, buf488, buf658, buf659, buf661, 128, 2048, grid=grid(128), stream=stream0)
        buf662 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_78, out1_79], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf657, buf658, buf659, primals_266, primals_267, buf662, 262144, grid=grid(262144), stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [out1_80], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, primals_268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf664 = buf659; del buf659  # reuse
        buf665 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf667 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_78], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf663, buf664, buf665, buf667, 128, 1024, grid=grid(128), stream=stream0)
        buf668 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_78, out2_79], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf663, buf664, buf665, primals_269, primals_270, buf668, 131072, grid=grid(131072), stream=stream0)
        del primals_270
        # Topologically Sorted Source Nodes: [out2_80], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf668, primals_271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf670 = buf665; del buf665  # reuse
        buf671 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf673 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_130], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf669, buf670, buf671, buf673, 128, 512, grid=grid(128), stream=stream0)
        buf674 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [out3_130, out3_131], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf669, buf670, buf671, primals_272, primals_273, buf674, 65536, grid=grid(65536), stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [out3_132], Original ATen: [aten.convolution]
        buf675 = extern_kernels.convolution(buf674, primals_274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf675, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf676 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_133, out3_134], Original ATen: [aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_43.run(buf663, buf669, buf675, buf657, buf676, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf677 = extern_kernels.convolution(buf676, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf677, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf678 = buf677; del buf677  # reuse
        buf679 = buf671; del buf671  # reuse
        buf680 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf682 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_92, group_norm_86], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_44.run(buf678, primals_276, buf679, buf680, buf682, 128, 2048, grid=grid(128), stream=stream0)
        del primals_276
        buf683 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_86, ll_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf678, buf679, buf680, primals_277, primals_278, buf683, 262144, grid=grid(262144), stream=stream0)
        del primals_278
        # Topologically Sorted Source Nodes: [tmp_out_2], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf685 = buf684; del buf684  # reuse
        # Topologically Sorted Source Nodes: [tmp_out_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_45.run(buf685, primals_280, 262144, grid=grid(262144), stream=stream0)
        del primals_280
        # Topologically Sorted Source Nodes: [ll_5], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf683, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [tmp_out__2], Original ATen: [aten.convolution]
        buf687 = extern_kernels.convolution(buf685, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf687, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf688 = buf686; del buf686  # reuse
        buf689 = buf680; del buf680  # reuse
        buf690 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf692 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [ll_5, tmp_out__2, add_10, previous_2, out1_81], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_46.run(buf688, buf488, primals_282, buf687, primals_284, buf689, buf690, buf692, 128, 2048, grid=grid(128), stream=stream0)
        del primals_282
        del primals_284
        buf693 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [out1_81, out1_82], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf688, buf689, buf690, primals_285, primals_286, buf693, 262144, grid=grid(262144), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [out1_83], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf693, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf695 = buf690; del buf690  # reuse
        buf696 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf698 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_81], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf694, buf695, buf696, buf698, 128, 1024, grid=grid(128), stream=stream0)
        buf699 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_81, out2_82], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf694, buf695, buf696, primals_288, primals_289, buf699, 131072, grid=grid(131072), stream=stream0)
        del primals_289
        # Topologically Sorted Source Nodes: [out2_83], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf701 = buf696; del buf696  # reuse
        buf702 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf704 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_135], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf700, buf701, buf702, buf704, 128, 512, grid=grid(128), stream=stream0)
        buf705 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [out3_135, out3_136], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf700, buf701, buf702, primals_291, primals_292, buf705, 65536, grid=grid(65536), stream=stream0)
        del primals_292
        # Topologically Sorted Source Nodes: [out3_137], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf707 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf708 = buf702; del buf702  # reuse
        buf709 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf711 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_6, out1_84], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_17.run(buf688, buf707, buf708, buf709, buf711, 128, 512, grid=grid(128), stream=stream0)
        buf712 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_84, out1_85], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf707, buf708, buf709, primals_294, primals_295, buf712, 65536, grid=grid(65536), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [out1_86], Original ATen: [aten.convolution]
        buf713 = extern_kernels.convolution(buf712, primals_296, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf713, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf714 = buf709; del buf709  # reuse
        buf715 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf717 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_84], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf713, buf714, buf715, buf717, 128, 256, grid=grid(128), stream=stream0)
        buf718 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_84, out2_85], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf713, buf714, buf715, primals_297, primals_298, buf718, 32768, grid=grid(32768), stream=stream0)
        del primals_298
        # Topologically Sorted Source Nodes: [out2_86], Original ATen: [aten.convolution]
        buf719 = extern_kernels.convolution(buf718, primals_299, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf719, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf720 = buf715; del buf715  # reuse
        buf721 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf723 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_140], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf719, buf720, buf721, buf723, 128, 128, grid=grid(128), stream=stream0)
        buf724 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [out3_140, out3_141], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf719, buf720, buf721, primals_300, primals_301, buf724, 16384, grid=grid(16384), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [out3_142], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf725, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf726 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf727 = buf721; del buf721  # reuse
        buf728 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf730 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_143, out3_144, out1_87], Original ATen: [aten.cat, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_23.run(buf713, buf719, buf725, buf707, buf726, buf727, buf728, buf730, 128, 512, grid=grid(128), stream=stream0)
        buf731 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_87, out1_88], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf726, buf727, buf728, primals_303, primals_304, buf731, 65536, grid=grid(65536), stream=stream0)
        del primals_304
        # Topologically Sorted Source Nodes: [out1_89], Original ATen: [aten.convolution]
        buf732 = extern_kernels.convolution(buf731, primals_305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf732, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf733 = buf728; del buf728  # reuse
        buf734 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf736 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_87], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf732, buf733, buf734, buf736, 128, 256, grid=grid(128), stream=stream0)
        buf737 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_87, out2_88], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf732, buf733, buf734, primals_306, primals_307, buf737, 32768, grid=grid(32768), stream=stream0)
        del primals_307
        # Topologically Sorted Source Nodes: [out2_89], Original ATen: [aten.convolution]
        buf738 = extern_kernels.convolution(buf737, primals_308, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf738, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf739 = buf734; del buf734  # reuse
        buf740 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf742 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_145], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf738, buf739, buf740, buf742, 128, 128, grid=grid(128), stream=stream0)
        buf743 = buf725; del buf725  # reuse
        # Topologically Sorted Source Nodes: [out3_145, out3_146], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf738, buf739, buf740, primals_309, primals_310, buf743, 16384, grid=grid(16384), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [out3_147], Original ATen: [aten.convolution]
        buf744 = extern_kernels.convolution(buf743, primals_311, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf744, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf745 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf746 = buf740; del buf740  # reuse
        buf747 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf749 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [low1_7, out1_90], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_24.run(buf726, buf745, buf746, buf747, buf749, 128, 128, grid=grid(128), stream=stream0)
        buf750 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_90, out1_91], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf745, buf746, buf747, primals_312, primals_313, buf750, 16384, grid=grid(16384), stream=stream0)
        del primals_313
        # Topologically Sorted Source Nodes: [out1_92], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, primals_314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf752 = buf747; del buf747  # reuse
        buf753 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf755 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_90], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf751, buf752, buf753, buf755, 128, 64, grid=grid(128), stream=stream0)
        buf756 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_90, out2_91], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf751, buf752, buf753, primals_315, primals_316, buf756, 8192, grid=grid(8192), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [out2_92], Original ATen: [aten.convolution]
        buf757 = extern_kernels.convolution(buf756, primals_317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf757, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf758 = buf753; del buf753  # reuse
        buf759 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf761 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_150], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf757, buf758, buf759, buf761, 128, 32, grid=grid(128), stream=stream0)
        buf762 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [out3_150, out3_151], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf757, buf758, buf759, primals_318, primals_319, buf762, 4096, grid=grid(4096), stream=stream0)
        del primals_319
        # Topologically Sorted Source Nodes: [out3_152], Original ATen: [aten.convolution]
        buf763 = extern_kernels.convolution(buf762, primals_320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf763, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf764 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf765 = buf759; del buf759  # reuse
        buf766 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf768 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_153, out1_93], Original ATen: [aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_group_norm_30.run(buf751, buf757, buf763, buf745, buf764, buf765, buf766, buf768, 128, 128, grid=grid(128), stream=stream0)
        buf769 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_93, out1_94], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_31.run(buf764, buf745, buf765, buf766, primals_321, primals_322, buf769, 16384, grid=grid(16384), stream=stream0)
        del primals_322
        # Topologically Sorted Source Nodes: [out1_95], Original ATen: [aten.convolution]
        buf770 = extern_kernels.convolution(buf769, primals_323, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf770, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf771 = buf766; del buf766  # reuse
        buf772 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf774 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_93], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf770, buf771, buf772, buf774, 128, 64, grid=grid(128), stream=stream0)
        buf775 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_93, out2_94], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf770, buf771, buf772, primals_324, primals_325, buf775, 8192, grid=grid(8192), stream=stream0)
        del primals_325
        # Topologically Sorted Source Nodes: [out2_95], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, primals_326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf776, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf777 = buf772; del buf772  # reuse
        buf778 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf780 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_155], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf776, buf777, buf778, buf780, 128, 32, grid=grid(128), stream=stream0)
        buf781 = buf763; del buf763  # reuse
        # Topologically Sorted Source Nodes: [out3_155, out3_156], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf776, buf777, buf778, primals_327, primals_328, buf781, 4096, grid=grid(4096), stream=stream0)
        del primals_328
        # Topologically Sorted Source Nodes: [out3_157], Original ATen: [aten.convolution]
        buf782 = extern_kernels.convolution(buf781, primals_329, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf782, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf783 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf784 = buf778; del buf778  # reuse
        buf785 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf787 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_154, out3_158, out3_159, out1_96], Original ATen: [aten.add, aten.cat, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_group_norm_32.run(buf770, buf776, buf782, buf764, buf745, buf783, buf784, buf785, buf787, 128, 128, grid=grid(128), stream=stream0)
        buf788 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_96, out1_97], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_25.run(buf783, buf784, buf785, primals_330, primals_331, buf788, 16384, grid=grid(16384), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [out1_98], Original ATen: [aten.convolution]
        buf789 = extern_kernels.convolution(buf788, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf789, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf790 = buf785; del buf785  # reuse
        buf791 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf793 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_96], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_26.run(buf789, buf790, buf791, buf793, 128, 64, grid=grid(128), stream=stream0)
        buf794 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_96, out2_97], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_27.run(buf789, buf790, buf791, primals_333, primals_334, buf794, 8192, grid=grid(8192), stream=stream0)
        del primals_334
        # Topologically Sorted Source Nodes: [out2_98], Original ATen: [aten.convolution]
        buf795 = extern_kernels.convolution(buf794, primals_335, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf795, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf796 = buf791; del buf791  # reuse
        buf797 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf799 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_160], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_28.run(buf795, buf796, buf797, buf799, 128, 32, grid=grid(128), stream=stream0)
        buf800 = buf782; del buf782  # reuse
        # Topologically Sorted Source Nodes: [out3_160, out3_161], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_29.run(buf795, buf796, buf797, primals_336, primals_337, buf800, 4096, grid=grid(4096), stream=stream0)
        del primals_337
        # Topologically Sorted Source Nodes: [out3_162], Original ATen: [aten.convolution]
        buf801 = extern_kernels.convolution(buf800, primals_338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf801, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf802 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf818 = buf802; del buf802  # reuse
        buf819 = buf818; del buf818  # reuse
        buf820 = buf819; del buf819  # reuse
        buf821 = buf797; del buf797  # reuse
        buf822 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf824 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2, out3_148, out3_149, out3_163, out3_164, up2_6, low2_3, out1_99], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_47.run(buf820, buf186, buf187, buf789, buf795, buf801, buf783, buf188, buf189, buf190, buf195, buf200, buf205, buf732, buf738, buf744, buf726, buf821, buf822, buf824, 128, 512, grid=grid(128), stream=stream0)
        del buf801
        buf825 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_99, out1_100], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf820, buf821, buf822, primals_339, primals_340, buf825, 65536, grid=grid(65536), stream=stream0)
        del primals_340
        # Topologically Sorted Source Nodes: [out1_101], Original ATen: [aten.convolution]
        buf826 = extern_kernels.convolution(buf825, primals_341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf826, (4, 128, 8, 8), (8192, 64, 8, 1))
        buf827 = buf822; del buf822  # reuse
        buf828 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf830 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_99], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_19.run(buf826, buf827, buf828, buf830, 128, 256, grid=grid(128), stream=stream0)
        buf831 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_99, out2_100], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_20.run(buf826, buf827, buf828, primals_342, primals_343, buf831, 32768, grid=grid(32768), stream=stream0)
        del primals_343
        # Topologically Sorted Source Nodes: [out2_101], Original ATen: [aten.convolution]
        buf832 = extern_kernels.convolution(buf831, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf832, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf833 = buf828; del buf828  # reuse
        buf834 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf836 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_165], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf832, buf833, buf834, buf836, 128, 128, grid=grid(128), stream=stream0)
        buf837 = buf744; del buf744  # reuse
        # Topologically Sorted Source Nodes: [out3_165, out3_166], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf832, buf833, buf834, primals_345, primals_346, buf837, 16384, grid=grid(16384), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [out3_167], Original ATen: [aten.convolution]
        buf838 = extern_kernels.convolution(buf837, primals_347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf838, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf839 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf855 = buf839; del buf839  # reuse
        buf856 = buf855; del buf855  # reuse
        buf857 = buf856; del buf856  # reuse
        buf858 = buf834; del buf834  # reuse
        buf859 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf861 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [up2_1, out3_138, out3_139, out3_168, out3_169, up2_7, hg_3, out1_102], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.add, aten.rsub, aten.cat, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_native_group_norm_rsub_sub_48.run(buf857, buf231, buf232, buf826, buf832, buf838, buf820, buf233, buf234, buf235, buf240, buf245, buf250, buf694, buf700, buf706, buf688, buf858, buf859, buf861, 128, 2048, grid=grid(128), stream=stream0)
        del buf838
        buf862 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1_102, out1_103], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf857, buf858, buf859, primals_348, primals_349, buf862, 262144, grid=grid(262144), stream=stream0)
        del primals_349
        # Topologically Sorted Source Nodes: [out1_104], Original ATen: [aten.convolution]
        buf863 = extern_kernels.convolution(buf862, primals_350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf863, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf864 = buf859; del buf859  # reuse
        buf865 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf867 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out2_102], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_14.run(buf863, buf864, buf865, buf867, 128, 1024, grid=grid(128), stream=stream0)
        buf868 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_102, out2_103], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_8.run(buf863, buf864, buf865, primals_351, primals_352, buf868, 131072, grid=grid(131072), stream=stream0)
        del primals_352
        # Topologically Sorted Source Nodes: [out2_104], Original ATen: [aten.convolution]
        buf869 = extern_kernels.convolution(buf868, primals_353, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf869, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf870 = buf865; del buf865  # reuse
        buf871 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf873 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out3_170], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf869, buf870, buf871, buf873, 128, 512, grid=grid(128), stream=stream0)
        buf874 = buf706; del buf706  # reuse
        # Topologically Sorted Source Nodes: [out3_170, out3_171], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf869, buf870, buf871, primals_354, primals_355, buf874, 65536, grid=grid(65536), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [out3_172], Original ATen: [aten.convolution]
        buf875 = extern_kernels.convolution(buf874, primals_356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf875, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf876 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_173, out3_174], Original ATen: [aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_43.run(buf863, buf869, buf875, buf857, buf876, 262144, grid=grid(262144), stream=stream0)
        del buf875
        # Topologically Sorted Source Nodes: [conv2d_120], Original ATen: [aten.convolution]
        buf877 = extern_kernels.convolution(buf876, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf877, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf878 = buf877; del buf877  # reuse
        buf879 = buf871; del buf871  # reuse
        buf880 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf882 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_120, group_norm_111], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_44.run(buf878, primals_358, buf879, buf880, buf882, 128, 2048, grid=grid(128), stream=stream0)
        del primals_358
        buf883 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_111, ll_6], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf878, buf879, buf880, primals_359, primals_360, buf883, 262144, grid=grid(262144), stream=stream0)
        del buf880
        del primals_360
        # Topologically Sorted Source Nodes: [tmp_out_3], Original ATen: [aten.convolution]
        buf884 = extern_kernels.convolution(buf883, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf884, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf885 = buf884; del buf884  # reuse
        # Topologically Sorted Source Nodes: [tmp_out_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_45.run(buf885, primals_362, 262144, grid=grid(262144), stream=stream0)
        del primals_362
    return (buf285, buf485, buf685, buf885, buf6, buf32, primals_1, primals_3, primals_4, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_63, primals_65, primals_66, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_89, primals_90, primals_92, primals_93, primals_95, primals_96, primals_98, primals_99, primals_101, primals_102, primals_104, primals_105, primals_107, primals_108, primals_110, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_288, primals_290, primals_291, primals_293, primals_294, primals_296, primals_297, primals_299, primals_300, primals_302, primals_303, primals_305, primals_306, primals_308, primals_309, primals_311, primals_312, primals_314, primals_315, primals_317, primals_318, primals_320, primals_321, primals_323, primals_324, primals_326, primals_327, primals_329, primals_330, primals_332, primals_333, primals_335, primals_336, primals_338, primals_339, primals_341, primals_342, primals_344, primals_345, primals_347, primals_348, primals_350, primals_351, primals_353, primals_354, primals_356, primals_357, primals_359, primals_361, buf1, reinterpret_tensor(buf2, (4, 32), (32, 1), 0), reinterpret_tensor(buf5, (4, 32), (32, 1), 0), buf6, reinterpret_tensor(buf7, (4, 32), (32, 1), 0), reinterpret_tensor(buf10, (4, 32), (32, 1), 0), buf11, buf12, reinterpret_tensor(buf13, (4, 32), (32, 1), 0), reinterpret_tensor(buf16, (4, 32), (32, 1), 0), buf17, buf18, reinterpret_tensor(buf19, (4, 32), (32, 1), 0), reinterpret_tensor(buf22, (4, 32), (32, 1), 0), buf23, reinterpret_tensor(buf25, (4, 32), (32, 1), 0), reinterpret_tensor(buf28, (4, 32), (32, 1), 0), buf29, buf31, buf32, reinterpret_tensor(buf33, (4, 32), (32, 1), 0), reinterpret_tensor(buf36, (4, 32), (32, 1), 0), buf37, buf38, reinterpret_tensor(buf39, (4, 32), (32, 1), 0), reinterpret_tensor(buf42, (4, 32), (32, 1), 0), buf43, buf44, reinterpret_tensor(buf45, (4, 32), (32, 1), 0), reinterpret_tensor(buf48, (4, 32), (32, 1), 0), buf49, buf51, reinterpret_tensor(buf52, (4, 32), (32, 1), 0), reinterpret_tensor(buf55, (4, 32), (32, 1), 0), buf56, buf57, reinterpret_tensor(buf58, (4, 32), (32, 1), 0), reinterpret_tensor(buf61, (4, 32), (32, 1), 0), buf62, buf63, reinterpret_tensor(buf64, (4, 32), (32, 1), 0), reinterpret_tensor(buf67, (4, 32), (32, 1), 0), buf68, buf70, buf72, reinterpret_tensor(buf73, (4, 32), (32, 1), 0), reinterpret_tensor(buf76, (4, 32), (32, 1), 0), buf77, buf78, reinterpret_tensor(buf79, (4, 32), (32, 1), 0), reinterpret_tensor(buf82, (4, 32), (32, 1), 0), buf83, buf84, reinterpret_tensor(buf85, (4, 32), (32, 1), 0), reinterpret_tensor(buf88, (4, 32), (32, 1), 0), buf89, buf91, reinterpret_tensor(buf92, (4, 32), (32, 1), 0), reinterpret_tensor(buf95, (4, 32), (32, 1), 0), buf96, buf97, reinterpret_tensor(buf98, (4, 32), (32, 1), 0), reinterpret_tensor(buf101, (4, 32), (32, 1), 0), buf102, buf103, reinterpret_tensor(buf104, (4, 32), (32, 1), 0), reinterpret_tensor(buf107, (4, 32), (32, 1), 0), buf108, buf110, reinterpret_tensor(buf111, (4, 32), (32, 1), 0), reinterpret_tensor(buf114, (4, 32), (32, 1), 0), buf115, buf116, reinterpret_tensor(buf117, (4, 32), (32, 1), 0), reinterpret_tensor(buf120, (4, 32), (32, 1), 0), buf121, buf122, reinterpret_tensor(buf123, (4, 32), (32, 1), 0), reinterpret_tensor(buf126, (4, 32), (32, 1), 0), buf127, buf129, reinterpret_tensor(buf130, (4, 32), (32, 1), 0), reinterpret_tensor(buf133, (4, 32), (32, 1), 0), buf134, buf135, reinterpret_tensor(buf136, (4, 32), (32, 1), 0), reinterpret_tensor(buf139, (4, 32), (32, 1), 0), buf140, buf141, reinterpret_tensor(buf142, (4, 32), (32, 1), 0), reinterpret_tensor(buf145, (4, 32), (32, 1), 0), buf146, buf148, reinterpret_tensor(buf149, (4, 32), (32, 1), 0), reinterpret_tensor(buf152, (4, 32), (32, 1), 0), buf153, buf154, reinterpret_tensor(buf155, (4, 32), (32, 1), 0), reinterpret_tensor(buf158, (4, 32), (32, 1), 0), buf159, buf160, reinterpret_tensor(buf161, (4, 32), (32, 1), 0), reinterpret_tensor(buf164, (4, 32), (32, 1), 0), buf165, buf167, reinterpret_tensor(buf168, (4, 32), (32, 1), 0), reinterpret_tensor(buf171, (4, 32), (32, 1), 0), buf172, buf173, reinterpret_tensor(buf174, (4, 32), (32, 1), 0), reinterpret_tensor(buf177, (4, 32), (32, 1), 0), buf178, buf179, reinterpret_tensor(buf180, (4, 32), (32, 1), 0), reinterpret_tensor(buf183, (4, 32), (32, 1), 0), buf184, buf186, buf187, buf188, buf189, buf190, buf195, buf200, buf205, buf212, reinterpret_tensor(buf213, (4, 32), (32, 1), 0), reinterpret_tensor(buf216, (4, 32), (32, 1), 0), buf217, buf218, reinterpret_tensor(buf219, (4, 32), (32, 1), 0), reinterpret_tensor(buf222, (4, 32), (32, 1), 0), buf223, buf224, reinterpret_tensor(buf225, (4, 32), (32, 1), 0), reinterpret_tensor(buf228, (4, 32), (32, 1), 0), buf229, buf231, buf232, buf233, buf234, buf235, buf240, buf245, buf250, buf257, reinterpret_tensor(buf258, (4, 32), (32, 1), 0), reinterpret_tensor(buf261, (4, 32), (32, 1), 0), buf262, buf263, reinterpret_tensor(buf264, (4, 32), (32, 1), 0), reinterpret_tensor(buf267, (4, 32), (32, 1), 0), buf268, buf269, reinterpret_tensor(buf270, (4, 32), (32, 1), 0), reinterpret_tensor(buf273, (4, 32), (32, 1), 0), buf274, buf276, buf278, reinterpret_tensor(buf279, (4, 32), (32, 1), 0), reinterpret_tensor(buf282, (4, 32), (32, 1), 0), buf283, buf285, buf288, reinterpret_tensor(buf289, (4, 32), (32, 1), 0), reinterpret_tensor(buf292, (4, 32), (32, 1), 0), buf293, buf294, reinterpret_tensor(buf295, (4, 32), (32, 1), 0), reinterpret_tensor(buf298, (4, 32), (32, 1), 0), buf299, buf300, reinterpret_tensor(buf301, (4, 32), (32, 1), 0), reinterpret_tensor(buf304, (4, 32), (32, 1), 0), buf305, buf307, reinterpret_tensor(buf308, (4, 32), (32, 1), 0), reinterpret_tensor(buf311, (4, 32), (32, 1), 0), buf312, buf313, reinterpret_tensor(buf314, (4, 32), (32, 1), 0), reinterpret_tensor(buf317, (4, 32), (32, 1), 0), buf318, buf319, reinterpret_tensor(buf320, (4, 32), (32, 1), 0), reinterpret_tensor(buf323, (4, 32), (32, 1), 0), buf324, buf326, reinterpret_tensor(buf327, (4, 32), (32, 1), 0), reinterpret_tensor(buf330, (4, 32), (32, 1), 0), buf331, buf332, reinterpret_tensor(buf333, (4, 32), (32, 1), 0), reinterpret_tensor(buf336, (4, 32), (32, 1), 0), buf337, buf338, reinterpret_tensor(buf339, (4, 32), (32, 1), 0), reinterpret_tensor(buf342, (4, 32), (32, 1), 0), buf343, buf345, reinterpret_tensor(buf346, (4, 32), (32, 1), 0), reinterpret_tensor(buf349, (4, 32), (32, 1), 0), buf350, buf351, reinterpret_tensor(buf352, (4, 32), (32, 1), 0), reinterpret_tensor(buf355, (4, 32), (32, 1), 0), buf356, buf357, reinterpret_tensor(buf358, (4, 32), (32, 1), 0), reinterpret_tensor(buf361, (4, 32), (32, 1), 0), buf362, buf364, reinterpret_tensor(buf365, (4, 32), (32, 1), 0), reinterpret_tensor(buf368, (4, 32), (32, 1), 0), buf369, buf370, reinterpret_tensor(buf371, (4, 32), (32, 1), 0), reinterpret_tensor(buf374, (4, 32), (32, 1), 0), buf375, buf376, reinterpret_tensor(buf377, (4, 32), (32, 1), 0), reinterpret_tensor(buf380, (4, 32), (32, 1), 0), buf381, buf383, reinterpret_tensor(buf384, (4, 32), (32, 1), 0), reinterpret_tensor(buf387, (4, 32), (32, 1), 0), buf388, buf389, reinterpret_tensor(buf390, (4, 32), (32, 1), 0), reinterpret_tensor(buf393, (4, 32), (32, 1), 0), buf394, buf395, reinterpret_tensor(buf396, (4, 32), (32, 1), 0), reinterpret_tensor(buf399, (4, 32), (32, 1), 0), buf400, buf420, reinterpret_tensor(buf421, (4, 32), (32, 1), 0), reinterpret_tensor(buf424, (4, 32), (32, 1), 0), buf425, buf426, reinterpret_tensor(buf427, (4, 32), (32, 1), 0), reinterpret_tensor(buf430, (4, 32), (32, 1), 0), buf431, buf432, reinterpret_tensor(buf433, (4, 32), (32, 1), 0), reinterpret_tensor(buf436, (4, 32), (32, 1), 0), buf437, buf457, reinterpret_tensor(buf458, (4, 32), (32, 1), 0), reinterpret_tensor(buf461, (4, 32), (32, 1), 0), buf462, buf463, reinterpret_tensor(buf464, (4, 32), (32, 1), 0), reinterpret_tensor(buf467, (4, 32), (32, 1), 0), buf468, buf469, reinterpret_tensor(buf470, (4, 32), (32, 1), 0), reinterpret_tensor(buf473, (4, 32), (32, 1), 0), buf474, buf476, buf478, reinterpret_tensor(buf479, (4, 32), (32, 1), 0), reinterpret_tensor(buf482, (4, 32), (32, 1), 0), buf483, buf485, buf488, reinterpret_tensor(buf489, (4, 32), (32, 1), 0), reinterpret_tensor(buf492, (4, 32), (32, 1), 0), buf493, buf494, reinterpret_tensor(buf495, (4, 32), (32, 1), 0), reinterpret_tensor(buf498, (4, 32), (32, 1), 0), buf499, buf500, reinterpret_tensor(buf501, (4, 32), (32, 1), 0), reinterpret_tensor(buf504, (4, 32), (32, 1), 0), buf505, buf507, reinterpret_tensor(buf508, (4, 32), (32, 1), 0), reinterpret_tensor(buf511, (4, 32), (32, 1), 0), buf512, buf513, reinterpret_tensor(buf514, (4, 32), (32, 1), 0), reinterpret_tensor(buf517, (4, 32), (32, 1), 0), buf518, buf519, reinterpret_tensor(buf520, (4, 32), (32, 1), 0), reinterpret_tensor(buf523, (4, 32), (32, 1), 0), buf524, buf526, reinterpret_tensor(buf527, (4, 32), (32, 1), 0), reinterpret_tensor(buf530, (4, 32), (32, 1), 0), buf531, buf532, reinterpret_tensor(buf533, (4, 32), (32, 1), 0), reinterpret_tensor(buf536, (4, 32), (32, 1), 0), buf537, buf538, reinterpret_tensor(buf539, (4, 32), (32, 1), 0), reinterpret_tensor(buf542, (4, 32), (32, 1), 0), buf543, buf545, reinterpret_tensor(buf546, (4, 32), (32, 1), 0), reinterpret_tensor(buf549, (4, 32), (32, 1), 0), buf550, buf551, reinterpret_tensor(buf552, (4, 32), (32, 1), 0), reinterpret_tensor(buf555, (4, 32), (32, 1), 0), buf556, buf557, reinterpret_tensor(buf558, (4, 32), (32, 1), 0), reinterpret_tensor(buf561, (4, 32), (32, 1), 0), buf562, buf564, reinterpret_tensor(buf565, (4, 32), (32, 1), 0), reinterpret_tensor(buf568, (4, 32), (32, 1), 0), buf569, buf570, reinterpret_tensor(buf571, (4, 32), (32, 1), 0), reinterpret_tensor(buf574, (4, 32), (32, 1), 0), buf575, buf576, reinterpret_tensor(buf577, (4, 32), (32, 1), 0), reinterpret_tensor(buf580, (4, 32), (32, 1), 0), buf581, buf583, reinterpret_tensor(buf584, (4, 32), (32, 1), 0), reinterpret_tensor(buf587, (4, 32), (32, 1), 0), buf588, buf589, reinterpret_tensor(buf590, (4, 32), (32, 1), 0), reinterpret_tensor(buf593, (4, 32), (32, 1), 0), buf594, buf595, reinterpret_tensor(buf596, (4, 32), (32, 1), 0), reinterpret_tensor(buf599, (4, 32), (32, 1), 0), buf600, buf620, reinterpret_tensor(buf621, (4, 32), (32, 1), 0), reinterpret_tensor(buf624, (4, 32), (32, 1), 0), buf625, buf626, reinterpret_tensor(buf627, (4, 32), (32, 1), 0), reinterpret_tensor(buf630, (4, 32), (32, 1), 0), buf631, buf632, reinterpret_tensor(buf633, (4, 32), (32, 1), 0), reinterpret_tensor(buf636, (4, 32), (32, 1), 0), buf637, buf657, reinterpret_tensor(buf658, (4, 32), (32, 1), 0), reinterpret_tensor(buf661, (4, 32), (32, 1), 0), buf662, buf663, reinterpret_tensor(buf664, (4, 32), (32, 1), 0), reinterpret_tensor(buf667, (4, 32), (32, 1), 0), buf668, buf669, reinterpret_tensor(buf670, (4, 32), (32, 1), 0), reinterpret_tensor(buf673, (4, 32), (32, 1), 0), buf674, buf676, buf678, reinterpret_tensor(buf679, (4, 32), (32, 1), 0), reinterpret_tensor(buf682, (4, 32), (32, 1), 0), buf683, buf685, buf688, reinterpret_tensor(buf689, (4, 32), (32, 1), 0), reinterpret_tensor(buf692, (4, 32), (32, 1), 0), buf693, buf694, reinterpret_tensor(buf695, (4, 32), (32, 1), 0), reinterpret_tensor(buf698, (4, 32), (32, 1), 0), buf699, buf700, reinterpret_tensor(buf701, (4, 32), (32, 1), 0), reinterpret_tensor(buf704, (4, 32), (32, 1), 0), buf705, buf707, reinterpret_tensor(buf708, (4, 32), (32, 1), 0), reinterpret_tensor(buf711, (4, 32), (32, 1), 0), buf712, buf713, reinterpret_tensor(buf714, (4, 32), (32, 1), 0), reinterpret_tensor(buf717, (4, 32), (32, 1), 0), buf718, buf719, reinterpret_tensor(buf720, (4, 32), (32, 1), 0), reinterpret_tensor(buf723, (4, 32), (32, 1), 0), buf724, buf726, reinterpret_tensor(buf727, (4, 32), (32, 1), 0), reinterpret_tensor(buf730, (4, 32), (32, 1), 0), buf731, buf732, reinterpret_tensor(buf733, (4, 32), (32, 1), 0), reinterpret_tensor(buf736, (4, 32), (32, 1), 0), buf737, buf738, reinterpret_tensor(buf739, (4, 32), (32, 1), 0), reinterpret_tensor(buf742, (4, 32), (32, 1), 0), buf743, buf745, reinterpret_tensor(buf746, (4, 32), (32, 1), 0), reinterpret_tensor(buf749, (4, 32), (32, 1), 0), buf750, buf751, reinterpret_tensor(buf752, (4, 32), (32, 1), 0), reinterpret_tensor(buf755, (4, 32), (32, 1), 0), buf756, buf757, reinterpret_tensor(buf758, (4, 32), (32, 1), 0), reinterpret_tensor(buf761, (4, 32), (32, 1), 0), buf762, buf764, reinterpret_tensor(buf765, (4, 32), (32, 1), 0), reinterpret_tensor(buf768, (4, 32), (32, 1), 0), buf769, buf770, reinterpret_tensor(buf771, (4, 32), (32, 1), 0), reinterpret_tensor(buf774, (4, 32), (32, 1), 0), buf775, buf776, reinterpret_tensor(buf777, (4, 32), (32, 1), 0), reinterpret_tensor(buf780, (4, 32), (32, 1), 0), buf781, buf783, reinterpret_tensor(buf784, (4, 32), (32, 1), 0), reinterpret_tensor(buf787, (4, 32), (32, 1), 0), buf788, buf789, reinterpret_tensor(buf790, (4, 32), (32, 1), 0), reinterpret_tensor(buf793, (4, 32), (32, 1), 0), buf794, buf795, reinterpret_tensor(buf796, (4, 32), (32, 1), 0), reinterpret_tensor(buf799, (4, 32), (32, 1), 0), buf800, buf820, reinterpret_tensor(buf821, (4, 32), (32, 1), 0), reinterpret_tensor(buf824, (4, 32), (32, 1), 0), buf825, buf826, reinterpret_tensor(buf827, (4, 32), (32, 1), 0), reinterpret_tensor(buf830, (4, 32), (32, 1), 0), buf831, buf832, reinterpret_tensor(buf833, (4, 32), (32, 1), 0), reinterpret_tensor(buf836, (4, 32), (32, 1), 0), buf837, buf857, reinterpret_tensor(buf858, (4, 32), (32, 1), 0), reinterpret_tensor(buf861, (4, 32), (32, 1), 0), buf862, buf863, reinterpret_tensor(buf864, (4, 32), (32, 1), 0), reinterpret_tensor(buf867, (4, 32), (32, 1), 0), buf868, buf869, reinterpret_tensor(buf870, (4, 32), (32, 1), 0), reinterpret_tensor(buf873, (4, 32), (32, 1), 0), buf874, buf876, buf878, reinterpret_tensor(buf879, (4, 32), (32, 1), 0), reinterpret_tensor(buf882, (4, 32), (32, 1), 0), buf883, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
