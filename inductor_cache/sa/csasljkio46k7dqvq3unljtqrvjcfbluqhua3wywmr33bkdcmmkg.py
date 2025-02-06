# AOT ID: ['7_inference']
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


# kernel path: inductor_cache/m4/cm4ktl5z67o7gi7rtkvnqlemtqtk4pb3pgn4ytqia7725dthzkt6.py
# Topologically Sorted Source Nodes: [mul, min_val, mul_1, min_val_1, mul_4, min_val_2, mul_5, min_val_3, observer_min_val, mul_2, max_val, mul_3, max_val_1, mul_6, max_val_2, mul_7, max_val_3, observer_max_val, truediv, sub, sign, abs_3, add_4, floor, output, clamp, add_5, output_1, truediv_3, sub_3, sign_2, abs_6, add_6, floor_1, output_2, clamp_1, add_7, output_3, output_4], Original ATen: [aten.mul, aten.min, aten.add, aten.minimum, aten.max, aten.maximum, aten.div, aten.sub, aten.sign, aten.abs, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   abs_3 => abs_3
#   abs_6 => abs_6
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   clamp => clamp_max, clamp_min
#   clamp_1 => clamp_max_1, clamp_min_1
#   floor => floor
#   floor_1 => floor_1
#   max_val => max_1
#   max_val_1 => add_1
#   max_val_2 => max_2
#   max_val_3 => add_3
#   min_val => min_1
#   min_val_1 => add
#   min_val_2 => min_2
#   min_val_3 => add_2
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   observer_max_val => maximum
#   observer_min_val => minimum
#   output => mul_8
#   output_1 => mul_9
#   output_2 => mul_10
#   output_3 => mul_11
#   output_4 => add_8
#   sign => sign
#   sign_2 => sign_1
#   sub => sub
#   sub_3 => sub_3
#   truediv => div
#   truediv_3 => div_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.9), kwargs = {})
#   %min_1 : [num_users=1] = call_function[target=torch.ops.aten.min.default](args = (%arg0_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%min_1, 0.1), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.9), kwargs = {})
#   %min_2 : [num_users=1] = call_function[target=torch.ops.aten.min.default](args = (%arg3_1,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%min_2, 0.1), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%add, %add_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.9), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%arg0_1,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%max_1, 0.1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, 0.9), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%arg3_1,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%max_2, 0.1), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%add_1, %add_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %arg6_1), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %arg7_1), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub,), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_3, 0.5), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add_4,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %floor), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul_8, %arg8_1), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min, %arg9_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, %arg7_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %arg6_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %arg6_1), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_3, %arg7_1), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_3,), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_6, 0.5), kwargs = {})
#   %floor_1 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add_6,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %floor_1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul_10, %arg8_1), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min_1, %arg9_1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, %arg7_1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %arg6_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_11), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %add_2), kwargs = {})
#   %copy__3 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %add_3), kwargs = {})
triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0 = async_compile.triton('triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr8': '*fp32', 'out_ptr10': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18), 'tt.equal_to': (17,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0', 'mutated_arg_names': ['in_ptr7', 'in_ptr9', 'out_ptr10', 'out_ptr8'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp6 = tl.load(in_ptr1 + (r0), None)
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.load(in_ptr3 + (0))
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp30 = tl.load(in_ptr4 + (0))
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.load(in_ptr5 + (0))
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp55 = tl.load(in_ptr6 + (0))
    tmp56 = tl.broadcast_to(tmp55, [1])
    tmp62 = tl.load(in_ptr7 + (0))
    tmp63 = tl.broadcast_to(tmp62, [1])
    tmp68 = tl.load(in_ptr8 + (0))
    tmp69 = tl.broadcast_to(tmp68, [1])
    tmp73 = tl.load(in_ptr9 + (0))
    tmp74 = tl.broadcast_to(tmp73, [1])
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp1, 0))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp7, 0))
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp14 = tmp0 / tmp13
    tmp17 = tmp14 - tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp18 < tmp17
    tmp20 = tmp19.to(tl.int8)
    tmp21 = tmp17 < tmp18
    tmp22 = tmp21.to(tl.int8)
    tmp23 = tmp20 - tmp22
    tmp24 = tmp23.to(tmp17.dtype)
    tmp25 = tl_math.abs(tmp17)
    tmp26 = 0.5
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.floor(tmp27)
    tmp29 = tmp24 * tmp28
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tmp35 = triton_helpers.minimum(tmp32, tmp34)
    tmp36 = tmp35 + tmp16
    tmp37 = tmp36 * tmp13
    tmp38 = tmp6 / tmp13
    tmp39 = tmp38 - tmp16
    tmp40 = tmp18 < tmp39
    tmp41 = tmp40.to(tl.int8)
    tmp42 = tmp39 < tmp18
    tmp43 = tmp42.to(tl.int8)
    tmp44 = tmp41 - tmp43
    tmp45 = tmp44.to(tmp39.dtype)
    tmp46 = tl_math.abs(tmp39)
    tmp47 = tmp46 + tmp26
    tmp48 = libdevice.floor(tmp47)
    tmp49 = tmp45 * tmp48
    tmp50 = triton_helpers.maximum(tmp49, tmp31)
    tmp51 = triton_helpers.minimum(tmp50, tmp34)
    tmp52 = tmp51 + tmp16
    tmp53 = tmp52 * tmp13
    tmp54 = tmp37 + tmp53
    tmp57 = 0.9
    tmp58 = tmp56 * tmp57
    tmp59 = 0.1
    tmp60 = tmp3 * tmp59
    tmp61 = tmp58 + tmp60
    tmp64 = tmp63 * tmp57
    tmp65 = tmp9 * tmp59
    tmp66 = tmp64 + tmp65
    tmp67 = triton_helpers.minimum(tmp61, tmp66)
    tmp70 = tmp69 * tmp57
    tmp71 = tmp5 * tmp59
    tmp72 = tmp70 + tmp71
    tmp75 = tmp74 * tmp57
    tmp76 = tmp11 * tmp59
    tmp77 = tmp75 + tmp76
    tmp78 = triton_helpers.maximum(tmp72, tmp77)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [RBLOCK])), tmp54, None)
    tl.store(out_ptr5 + (tl.full([1], 0, tl.int32)), tmp67, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp78, None)
    tl.store(out_ptr8 + (tl.full([1], 0, tl.int32)), tmp66, None)
    tl.store(out_ptr10 + (tl.full([1], 0, tl.int32)), tmp77, None)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf74kcy2dsdlho3ka72ulfasqo6uesrygcslqwyv2myupgd5xfos.py
# Topologically Sorted Source Nodes: [mul, mul_1, min_val_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   min_val_1 => add
#   mul => mul
#   mul_1 => mul_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.9), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%min_1, 0.1), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add), kwargs = {})
triton_poi_fused_add_mul_1 = async_compile.triton('triton_poi_fused_add_mul_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_1', 'mutated_arg_names': ['in_out_ptr0', 'in_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp2 = 0.9
    tmp3 = tmp1 * tmp2
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 + tmp7
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (1, ), (1, ))
    assert_size_stride(arg3_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg4_1, (1, ), (1, ))
    assert_size_stride(arg5_1, (1, ), (1, ))
    assert_size_stride(arg6_1, (1, ), (1, ))
    assert_size_stride(arg7_1, (1, ), (1, ))
    assert_size_stride(arg8_1, (), ())
    assert_size_stride(arg9_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf7 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf11 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, min_val, mul_1, min_val_1, mul_4, min_val_2, mul_5, min_val_3, observer_min_val, mul_2, max_val, mul_3, max_val_1, mul_6, max_val_2, mul_7, max_val_3, observer_max_val, truediv, sub, sign, abs_3, add_4, floor, output, clamp, add_5, output_1, truediv_3, sub_3, sign_2, abs_6, add_6, floor_1, output_2, clamp_1, add_7, output_3, output_4], Original ATen: [aten.mul, aten.min, aten.add, aten.minimum, aten.max, aten.maximum, aten.div, aten.sub, aten.sign, aten.abs, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0.run(arg0_1, arg3_1, arg6_1, arg7_1, arg8_1, arg9_1, arg1_1, arg4_1, arg2_1, arg5_1, buf0, buf2, buf4, buf7, buf11, arg4_1, arg5_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf6 = reinterpret_tensor(buf0, (1, ), (1, ), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul, mul_1, min_val_1], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_1.run(buf6, arg1_1, arg1_1, 1, grid=grid(1), stream=stream0)
        del arg1_1
        del buf6
        buf10 = reinterpret_tensor(buf2, (1, ), (1, ), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul_2, mul_3, max_val_1], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_1.run(buf10, arg2_1, arg2_1, 1, grid=grid(1), stream=stream0)
        del arg2_1
        del buf10
    return (buf4, buf7, buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
