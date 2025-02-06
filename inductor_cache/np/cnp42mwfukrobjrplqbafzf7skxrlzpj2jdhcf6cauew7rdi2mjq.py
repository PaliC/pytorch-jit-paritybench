# AOT ID: ['10_forward']
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


# kernel path: inductor_cache/yc/cyc3alpuaetjs733jm7t7w2nk55igron4uqko7tu3aaqh2im2k46.py
# Topologically Sorted Source Nodes: [truediv, sub, sign, abs_3, add, floor, output, clamp, add_1, output_1], Original ATen: [aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.mul, aten.clamp]
# Source node to ATen node mapping:
#   abs_3 => abs_3
#   add => add
#   add_1 => add_1
#   clamp => clamp_max, clamp_min
#   floor => floor
#   output => mul
#   output_1 => mul_1
#   sign => sign
#   sub => sub
#   truediv => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %primals_3), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub,), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_3, 0.5), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %floor), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul, %primals_6), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min, %primals_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, %primals_3), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %primals_1), kwargs = {})
triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr4 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tmp6 = tmp3 - tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 < tmp6
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp6 < tmp7
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tmp6.dtype)
    tmp14 = tl_math.abs(tmp6)
    tmp15 = 0.5
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp13 * tmp17
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp24 = triton_helpers.minimum(tmp21, tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = tmp25 * tmp2
    tl.store(out_ptr0 + (x0), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4l/c4l7l4bdnbq326sflrzny7g6rveeysg6ntddfudgfljdssmdl6n5.py
# Topologically Sorted Source Nodes: [truediv_3, sub_3, truediv_4, sub_4, truediv_5, sub_5, output_2, clamp_1, add_2, output_3], Original ATen: [aten.div, aten.sub, aten.abs, aten.maximum, aten.neg, aten.sign, aten.add, aten.floor, aten.mul, aten.clamp, aten.ge, aten.le, aten.logical_and, aten.gt, aten.lt]
# Source node to ATen node mapping:
#   add_2 => add_3
#   clamp_1 => clamp_max_1, clamp_min_1
#   output_2 => abs_4, abs_5, abs_6, add_2, floor_1, maximum_1, mul_2, neg_1, sign_1
#   output_3 => mul_3
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
# Graph fragment:
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_8, %primals_9), kwargs = {})
#   %sub_3 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_3, %primals_10), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_11, %primals_9), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %primals_10), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_12, %primals_9), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_5, %primals_10), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %maximum_1 : [num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_4, %abs_5), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%maximum_1,), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_3,), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_6, 0.5), kwargs = {})
#   %floor_1 : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%add_2,), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %floor_1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%mul_2, %primals_13), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min_1, %primals_14), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, %primals_10), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %primals_9), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%mul_2, %primals_13), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%mul_2, %primals_14), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%sub_3, %maximum_1), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%sub_3, %neg_1), kwargs = {})
triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i1', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp28 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tmp5 < tmp4
    tmp7 = tmp6.to(tl.int8)
    tmp8 = tmp4 < tmp5
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tmp4.dtype)
    tmp12 = tl_math.abs(tmp4)
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = tmp11 * tmp15
    tmp19 = triton_helpers.maximum(tmp16, tmp18)
    tmp22 = triton_helpers.minimum(tmp19, tmp21)
    tmp23 = tmp22 + tmp3
    tmp24 = tmp23 * tmp1
    tmp25 = tmp16 >= tmp18
    tmp26 = tmp16 <= tmp21
    tmp27 = tmp25 & tmp26
    tmp29 = tmp28 / tmp1
    tmp30 = tmp29 - tmp3
    tmp31 = tl_math.abs(tmp30)
    tmp33 = tmp32 / tmp1
    tmp34 = tmp33 - tmp3
    tmp35 = tl_math.abs(tmp34)
    tmp36 = triton_helpers.maximum(tmp31, tmp35)
    tmp37 = tmp4 > tmp36
    tmp38 = -tmp36
    tmp39 = tmp4 < tmp38
    tl.store(out_ptr0 + (x2), tmp24, xmask)
    tl.store(out_ptr1 + (x2), tmp27, xmask)
    tl.store(out_ptr2 + (x2), tmp37, xmask)
    tl.store(out_ptr3 + (x2), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tm/ctm5fh7ijf4dp5bu4sjze3ylrq73qznnz5a55atvan2a2qthkkwi.py
# Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   output_4 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_1, %mul_3, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (1, ), (1, ))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (1, ), (1, ))
    assert_size_stride(primals_4, (1, ), (1, ))
    assert_size_stride(primals_5, (1, ), (1, ))
    assert_size_stride(primals_6, (), ())
    assert_size_stride(primals_7, (), ())
    assert_size_stride(primals_8, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_9, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_11, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_12, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (), ())
    assert_size_stride(primals_14, (), ())
    assert_size_stride(primals_15, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv, sub, sign, abs_3, add, floor, output, clamp, add_1, output_1], Original ATen: [aten.div, aten.sub, aten.sign, aten.abs, aten.add, aten.floor, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_floor_mul_sign_sub_0.run(primals_2, primals_1, primals_3, primals_6, primals_7, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_6
        del primals_7
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [truediv_3, sub_3, truediv_4, sub_4, truediv_5, sub_5, output_2, clamp_1, add_2, output_3], Original ATen: [aten.div, aten.sub, aten.abs, aten.maximum, aten.neg, aten.sign, aten.add, aten.floor, aten.mul, aten.clamp, aten.ge, aten.le, aten.logical_and, aten.gt, aten.lt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1.run(primals_8, primals_9, primals_10, primals_13, primals_14, primals_11, primals_12, buf1, buf4, buf5, buf6, 256, grid=grid(256), stream=stream0)
        del primals_10
        del primals_11
        del primals_12
        del primals_13
        del primals_14
        del primals_8
        # Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 1, 1), (4, 1, 1, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [output_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf3, primals_15, 16, grid=grid(16), stream=stream0)
        del primals_15
    return (buf3, primals_9, buf0, buf1, buf4, buf5, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
