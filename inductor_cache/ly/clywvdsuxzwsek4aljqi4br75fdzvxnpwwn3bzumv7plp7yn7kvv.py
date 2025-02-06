# AOT ID: ['10_inference']
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


# kernel path: inductor_cache/ru/cruukk6v2it7zysmze2n627e4tq44akfzegyzyqj4vwgts5q364w.py
# Topologically Sorted Source Nodes: [argmax, eq, sum_1, argmax_1, eq_1, sum_2, argmax_2, eq_2, sum_3, argmax_3, eq_3, sum_4, durations], Original ATen: [aten.argmax, aten.eq, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   argmax => argmax
#   argmax_1 => argmax_1
#   argmax_2 => argmax_2
#   argmax_3 => argmax_3
#   durations => cat
#   eq => eq
#   eq_1 => eq_1
#   eq_2 => eq_2
#   eq_3 => eq_3
#   sum_1 => sum_1
#   sum_2 => sum_2
#   sum_3 => sum_3
#   sum_4 => sum_4
# Graph fragment:
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%arg0_1, -1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%argmax, 0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq,), kwargs = {})
#   %argmax_1 : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%arg0_1, -1), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%argmax_1, 1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq_1,), kwargs = {})
#   %argmax_2 : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%arg0_1, -1), kwargs = {})
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%argmax_2, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq_2,), kwargs = {})
#   %argmax_3 : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%arg0_1, -1), kwargs = {})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%argmax_3, 3), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq_3,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3],), kwargs = {})
triton_per_fused_argmax_eq_stack_sum_0 = async_compile.triton('triton_per_fused_argmax_eq_stack_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr8': '*i64', 'out_ptr9': '*i64', 'out_ptr10': '*i64', 'out_ptr11': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_argmax_eq_stack_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_argmax_eq_stack_sum_0(in_ptr0, out_ptr8, out_ptr9, out_ptr10, out_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1, 1], 0, tl.int64)
    tmp11 = tl.full([1, 1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 > tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1, 1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 > tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1, 1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tmp47 = tmp46 == tmp10
    tmp48 = tmp47.to(tl.int64)
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.sum(tmp49, 1)[:, None]
    tmp52 = tmp46 == tmp11
    tmp53 = tmp52.to(tl.int64)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.sum(tmp54, 1)[:, None]
    tmp57 = tmp46 == tmp26
    tmp58 = tmp57.to(tl.int64)
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
    tmp61 = tl.sum(tmp59, 1)[:, None]
    tmp62 = tmp46 == tmp41
    tmp63 = tmp62.to(tl.int64)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp66 = tl.sum(tmp64, 1)[:, None]
    tl.store(out_ptr8 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp66, None)
    tl.store(out_ptr9 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp61, None)
    tl.store(out_ptr10 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp56, None)
    tl.store(out_ptr11 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp51, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf12 = empty_strided_cuda((4, ), (1, ), torch.int64)
        buf11 = reinterpret_tensor(buf12, (1, ), (1, ), 3)  # alias
        buf10 = reinterpret_tensor(buf12, (1, ), (1, ), 2)  # alias
        buf9 = reinterpret_tensor(buf12, (1, ), (1, ), 1)  # alias
        buf8 = reinterpret_tensor(buf12, (1, ), (1, ), 0)  # alias
        # Topologically Sorted Source Nodes: [argmax, eq, sum_1, argmax_1, eq_1, sum_2, argmax_2, eq_2, sum_3, argmax_3, eq_3, sum_4, durations], Original ATen: [aten.argmax, aten.eq, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_argmax_eq_stack_sum_0.run(arg0_1, buf11, buf10, buf9, buf8, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
    return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
