# AOT ID: ['14_inference']
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


# kernel path: inductor_cache/gz/cgzuw4f24pbudqi2th5fvkcxwmbysaott7ajeqcvhq4gskyx6wzw.py
# Topologically Sorted Source Nodes: [input_3, target_1, mul_2, a, mul_3, sum_2, mul_4, sum_3], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   a => sum_1
#   input_3 => mul
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   sum_2 => sum_2
#   sum_3 => sum_3
#   target_1 => mul_1
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view_2), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %view_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %mul), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1]), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %mul_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1]), kwargs = {})
triton_per_fused_mul_sum_0 = async_compile.triton('triton_per_fused_mul_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp2 = tl.load(in_ptr1 + (r1 + 64*x0), xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4 * tmp2
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp3 * tmp3
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp5 * tmp5
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bv/cbvzkjankv6hpgm4muocirwwfupqy2q4muusiog7ggosh5bno44o.py
# Topologically Sorted Source Nodes: [mul_5, b, c, add_2, d, loss, loss_1, loss_2], Original ATen: [aten.mul, aten.add, aten.div, aten.rsub, aten.mean]
# Source node to ATen node mapping:
#   add_2 => add_2
#   b => add
#   c => add_1
#   d => div
#   loss => sub
#   loss_1 => mul_6
#   loss_2 => mean
#   mul_5 => mul_5
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 0.001), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 0.001), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %add_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_5, %add_2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 1.0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_6,), kwargs = {})
triton_poi_fused_add_div_mean_mul_rsub_1 = async_compile.triton('triton_poi_fused_add_div_mean_mul_rsub_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr2 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp30 = tl.load(in_ptr0 + (2))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (2))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp36 = tl.load(in_ptr2 + (2))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp44 = tl.load(in_ptr0 + (3))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr1 + (3))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp50 = tl.load(in_ptr2 + (3))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp10 = tmp9 + tmp6
    tmp11 = tmp7 + tmp10
    tmp12 = tmp3 / tmp11
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp14 * tmp13
    tmp18 = tmp17 * tmp2
    tmp21 = tmp20 + tmp6
    tmp24 = tmp23 + tmp6
    tmp25 = tmp21 + tmp24
    tmp26 = tmp18 / tmp25
    tmp27 = tmp13 - tmp26
    tmp28 = tmp27 * tmp13
    tmp29 = tmp15 + tmp28
    tmp32 = tmp31 * tmp2
    tmp35 = tmp34 + tmp6
    tmp38 = tmp37 + tmp6
    tmp39 = tmp35 + tmp38
    tmp40 = tmp32 / tmp39
    tmp41 = tmp13 - tmp40
    tmp42 = tmp41 * tmp13
    tmp43 = tmp29 + tmp42
    tmp46 = tmp45 * tmp2
    tmp49 = tmp48 + tmp6
    tmp52 = tmp51 + tmp6
    tmp53 = tmp49 + tmp52
    tmp54 = tmp46 / tmp53
    tmp55 = tmp13 - tmp54
    tmp56 = tmp55 * tmp13
    tmp57 = tmp43 + tmp56
    tmp58 = 4.0
    tmp59 = tmp57 / tmp58
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp59, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, target_1, mul_2, a, mul_3, sum_2, mul_4, sum_3], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_0.run(arg0_1, arg2_1, arg1_1, buf0, buf1, buf2, 4, 64, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mul_5, b, c, add_2, d, loss, loss_1, loss_2], Original ATen: [aten.mul, aten.add, aten.div, aten.rsub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_rsub_1.run(buf0, buf1, buf2, buf3, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
        del buf2
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
