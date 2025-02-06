# AOT ID: ['20_inference']
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


# kernel path: inductor_cache/i7/ci7vbgwqycerfwvegbqxudlcafl5a7r5aud4d752x4l6hap5v5ou.py
# Topologically Sorted Source Nodes: [mul, tp, sub, mul_1, fp, sub_1, mul_2, fn], Original ATen: [aten.mul, aten.sum, aten.rsub]
# Source node to ATen node mapping:
#   fn => sum_5
#   fp => sum_4
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   sub => sub_1
#   sub_1 => sub_2
#   tp => sum_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [0, 2, 3]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg1_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %sub_1), kwargs = {})
#   %sum_4 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0, 2, 3]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %sub_2), kwargs = {})
#   %sum_5 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [0, 2, 3]), kwargs = {})
triton_per_fused_mul_rsub_sum_0 = async_compile.triton('triton_per_fused_mul_rsub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_rsub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_rsub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = (rindex % 16)
    r2 = rindex // 16
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0 + 64*r2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16*x0 + 64*r2), xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 1.0
    tmp8 = tmp7 - tmp1
    tmp9 = tmp0 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp7 - tmp0
    tmp15 = tmp1 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tl.store(out_ptr2 + (x0), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hi/chiri6bq6n2nclomidzhgxhxkahwrarkqvydk6wmqefi74yxb4up.py
# Topologically Sorted Source Nodes: [mul_3, add_1, union, eq, float_1, mul_4, add_3, mul_5, add_4, add_5, add_6, score, metric, sub_2], Original ATen: [aten.mul, aten.add, aten.eq, aten._to_copy, aten.div, aten.mean, aten.rsub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   eq => eq
#   float_1 => convert_element_type
#   metric => mean
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   score => div
#   sub_2 => sub_3
#   union => add_2
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %sum_5), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_2, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1e-07), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %sum_4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %sum_5), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, 1e-07), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_3, %add_6), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%div,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mean), kwargs = {})
triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (1))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (1))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (1))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (2))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp41 = tl.load(in_ptr1 + (2))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr2 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp56 = tl.load(in_ptr0 + (3))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp59 = tl.load(in_ptr1 + (3))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp62 = tl.load(in_ptr2 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp1 + tmp5
    tmp9 = tmp6 + tmp8
    tmp10 = 0.0
    tmp11 = tmp9 == tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1e-07
    tmp14 = tmp12 * tmp13
    tmp15 = tmp3 + tmp14
    tmp16 = tmp3 + tmp5
    tmp17 = tmp16 + tmp8
    tmp18 = tmp17 + tmp13
    tmp19 = tmp15 / tmp18
    tmp22 = tmp21 * tmp2
    tmp25 = tmp21 + tmp24
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 == tmp10
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp13
    tmp32 = tmp22 + tmp31
    tmp33 = tmp22 + tmp24
    tmp34 = tmp33 + tmp27
    tmp35 = tmp34 + tmp13
    tmp36 = tmp32 / tmp35
    tmp37 = tmp19 + tmp36
    tmp40 = tmp39 * tmp2
    tmp43 = tmp39 + tmp42
    tmp46 = tmp43 + tmp45
    tmp47 = tmp46 == tmp10
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp13
    tmp50 = tmp40 + tmp49
    tmp51 = tmp40 + tmp42
    tmp52 = tmp51 + tmp45
    tmp53 = tmp52 + tmp13
    tmp54 = tmp50 / tmp53
    tmp55 = tmp37 + tmp54
    tmp58 = tmp57 * tmp2
    tmp61 = tmp57 + tmp60
    tmp64 = tmp61 + tmp63
    tmp65 = tmp64 == tmp10
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66 * tmp13
    tmp68 = tmp58 + tmp67
    tmp69 = tmp58 + tmp60
    tmp70 = tmp69 + tmp63
    tmp71 = tmp70 + tmp13
    tmp72 = tmp68 / tmp71
    tmp73 = tmp55 + tmp72
    tmp74 = 4.0
    tmp75 = tmp73 / tmp74
    tmp76 = 1.0
    tmp77 = tmp76 - tmp75
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp77, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, tp, sub, mul_1, fp, sub_1, mul_2, fn], Original ATen: [aten.mul, aten.sum, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_rsub_sum_0.run(arg0_1, arg1_1, buf0, buf1, buf2, 4, 64, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mul_3, add_1, union, eq, float_1, mul_4, add_3, mul_5, add_4, add_5, add_6, score, metric, sub_2], Original ATen: [aten.mul, aten.add, aten.eq, aten._to_copy, aten.div, aten.mean, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1.run(buf0, buf1, buf2, buf3, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
        del buf2
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
