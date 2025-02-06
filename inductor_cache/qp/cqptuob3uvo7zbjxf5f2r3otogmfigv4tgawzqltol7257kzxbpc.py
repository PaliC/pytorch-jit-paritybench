# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/4z/c4z2dvnjr3qeyeqem5qsp6kjtdl2crrzpvny2mt4xnfuvx2fzs65.py
# Topologically Sorted Source Nodes: [eq_1, masked_fill_, predicts, corrects, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.argmax, aten.sum]
# Source node to ATen node mapping:
#   corrects => eq
#   eq_1 => eq_1
#   masked_fill_ => full_default, where
#   predicts => argmax
#   sum_1 => sum_1
# Graph fragment:
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%view, 1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%argmax, %arg1_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default, %eq), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where,), kwargs = {})
triton_per_fused_argmax_eq_masked_fill_sum_0 = async_compile.triton('triton_per_fused_argmax_eq_masked_fill_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_argmax_eq_masked_fill_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_argmax_eq_masked_fill_sum_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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
    tmp47 = tl.load(in_ptr1 + (r0), None)
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
    tmp48 = 0.0
    tmp49 = tmp47 == tmp48
    tmp50 = tmp46.to(tl.float32)
    tmp51 = tmp50 == tmp47
    tmp52 = tl.full([1, 1], False, tl.int1)
    tmp53 = tl.where(tmp49, tmp52, tmp51)
    tmp54 = tmp53.to(tl.int64)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp57 = tl.sum(tmp55, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp57, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(arg1_1, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.int64)
        # Topologically Sorted Source Nodes: [eq_1, masked_fill_, predicts, corrects, sum_1], Original ATen: [aten.eq, aten.masked_fill, aten.argmax, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_argmax_eq_masked_fill_sum_0.run(arg0_1, arg1_1, buf1, 1, 16, grid=grid(1), stream=stream0)
        del arg0_1
    return (buf1, arg1_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
