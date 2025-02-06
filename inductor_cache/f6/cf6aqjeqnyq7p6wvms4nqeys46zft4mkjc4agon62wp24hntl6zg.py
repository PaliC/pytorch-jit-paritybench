# AOT ID: ['21_forward']
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


# kernel path: inductor_cache/bv/cbvirxij63zjh4hhj4vfztwfd6iipesbo34q4cnsf4td6e5fnnyy.py
# Topologically Sorted Source Nodes: [add, sqrt, add_2, sqrt_1, ratio_s, ratio_t, sub_1, dist, add_4, dist_inv, sum_1], Original ATen: [aten.add, aten.sqrt, aten.div, aten.sub, aten.abs, aten.reciprocal, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_4 => add_4
#   dist => abs_1
#   dist_inv => mul_1, reciprocal
#   ratio_s => div_1
#   ratio_t => div_2
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sub_1 => sub_1
#   sum_1 => sum_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, 1e-05), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, 1e-05), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_4, %sqrt_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %sqrt), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %div_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_4,), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0 = async_compile.triton('triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp29 = tl.load(in_ptr2 + (1))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp31 = tl.load(in_ptr3 + (1))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp49 = tl.load(in_ptr2 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp51 = tl.load(in_ptr3 + (2))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp62 = tl.load(in_ptr0 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp64 = tl.load(in_ptr1 + (3))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp69 = tl.load(in_ptr2 + (3))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp71 = tl.load(in_ptr3 + (3))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tmp1 / tmp6
    tmp12 = tmp11 + tmp4
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp9 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tl_math.abs(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = tmp20 * tmp17
    tmp26 = tmp25 + tmp4
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp23 / tmp27
    tmp33 = tmp32 + tmp4
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp30 / tmp34
    tmp36 = tmp28 - tmp35
    tmp37 = tl_math.abs(tmp36)
    tmp38 = tmp37 + tmp17
    tmp39 = tmp19 / tmp38
    tmp40 = tmp39 * tmp17
    tmp41 = tmp21 + tmp40
    tmp46 = tmp45 + tmp4
    tmp47 = libdevice.sqrt(tmp46)
    tmp48 = tmp43 / tmp47
    tmp53 = tmp52 + tmp4
    tmp54 = libdevice.sqrt(tmp53)
    tmp55 = tmp50 / tmp54
    tmp56 = tmp48 - tmp55
    tmp57 = tl_math.abs(tmp56)
    tmp58 = tmp57 + tmp17
    tmp59 = tmp19 / tmp58
    tmp60 = tmp59 * tmp17
    tmp61 = tmp41 + tmp60
    tmp66 = tmp65 + tmp4
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = tmp63 / tmp67
    tmp73 = tmp72 + tmp4
    tmp74 = libdevice.sqrt(tmp73)
    tmp75 = tmp70 / tmp74
    tmp76 = tmp68 - tmp75
    tmp77 = tl_math.abs(tmp76)
    tmp78 = tmp77 + tmp17
    tmp79 = tmp19 / tmp78
    tmp80 = tmp79 * tmp17
    tmp81 = tmp61 + tmp80
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp81, None)
''', device_str='cuda')


# kernel path: inductor_cache/3v/c3vvmscadwzhokatutmmp3ddnayi6jwiv5kewbp35dlkijwhdtq7.py
# Topologically Sorted Source Nodes: [add, sqrt, add_2, sqrt_1, ratio_s, ratio_t, sub_1, dist, add_4, dist_inv, mul_1, sum_1, alpha, add_5], Original ATen: [aten.add, aten.sqrt, aten.div, aten.sub, aten.abs, aten.reciprocal, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_4 => add_4
#   add_5 => add_5
#   alpha => div_3
#   dist => abs_1
#   dist_inv => mul_1, reciprocal
#   mul_1 => mul_2
#   ratio_s => div_1
#   ratio_t => div_2
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sub_1 => sub_1
#   sum_1 => sum_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, 1e-05), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, 1e-05), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_4, %sqrt_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %sqrt), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %div_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_4,), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 4), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %sum_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, 1), kwargs = {})
triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_1 = async_compile.triton('triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = tmp0 / tmp4
    tmp8 = tmp7 + tmp2
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = tmp6 / tmp9
    tmp11 = tmp5 - tmp10
    tmp12 = tl_math.abs(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = 4.0
    tmp19 = tmp17 * tmp18
    tmp22 = tmp19 / tmp21
    tmp23 = tmp22 + tmp13
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2jer2m3l5yn2huukqv64igmbgaccwhu2syvkxkuajxbzhl3kcz3.py
# Topologically Sorted Source Nodes: [sub, add, sqrt, output, mul, output_1, output_2], Original ATen: [aten.sub, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
#   output => div
#   output_1 => add_1
#   output_2 => mul_3
#   sqrt => sqrt
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %view_2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, 1e-05), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %view_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %add_1), kwargs = {})
triton_poi_fused_add_div_mul_sqrt_sub_2 = async_compile.triton('triton_poi_fused_add_div_mul_sqrt_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sqrt_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sqrt_sub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tmp3 / tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp0 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [add, sqrt, add_2, sqrt_1, ratio_s, ratio_t, sub_1, dist, add_4, dist_inv, sum_1], Original ATen: [aten.add, aten.sqrt, aten.div, aten.sub, aten.abs, aten.reciprocal, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0.run(primals_6, primals_7, primals_4, primals_5, buf0, 1, grid=grid(1), stream=stream0)
        buf1 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, sqrt, add_2, sqrt_1, ratio_s, ratio_t, sub_1, dist, add_4, dist_inv, mul_1, sum_1, alpha, add_5], Original ATen: [aten.add, aten.sqrt, aten.div, aten.sub, aten.abs, aten.reciprocal, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_1.run(primals_6, primals_7, primals_4, primals_5, buf0, buf1, 4, grid=grid(4), stream=stream0)
        del buf0
        del primals_6
        del primals_7
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, add, sqrt, output, mul, output_1, output_2], Original ATen: [aten.sub, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sqrt_sub_2.run(buf1, primals_1, primals_4, primals_5, primals_2, primals_3, buf2, 256, grid=grid(256), stream=stream0)
        del primals_2
        del primals_3
    return (buf2, primals_1, primals_4, primals_5, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
