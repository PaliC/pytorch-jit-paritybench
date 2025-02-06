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


# kernel path: inductor_cache/sw/cswhefg2vohlf7sa7ii2bgib5ibomafvasgyzpazdmzpgsgbwxn3.py
# Topologically Sorted Source Nodes: [neg, add, log, mul, policy_error, add_1, total_error], Original ATen: [aten.neg, aten.add, aten.log, aten.mul, aten.sum, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   log => log
#   mul => mul
#   neg => neg
#   policy_error => sum_1
#   total_error => mean
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg2_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-06), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %log), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, %sum_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_1,), kwargs = {})
triton_per_fused_add_log_mean_mul_neg_sum_0 = async_compile.triton('triton_per_fused_add_log_mean_mul_neg_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_log_mean_mul_neg_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_log_mean_mul_neg_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp4 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = -tmp4
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = tl_math.log(tmp8)
    tmp10 = tmp5 * tmp9
    tmp12 = -tmp11
    tmp14 = tmp13 + tmp7
    tmp15 = tl_math.log(tmp14)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp10 + tmp16
    tmp19 = -tmp18
    tmp21 = tmp20 + tmp7
    tmp22 = tl_math.log(tmp21)
    tmp23 = tmp19 * tmp22
    tmp24 = tmp17 + tmp23
    tmp26 = -tmp25
    tmp28 = tmp27 + tmp7
    tmp29 = tl_math.log(tmp28)
    tmp30 = tmp26 * tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp3 + tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.sum(tmp33, 1)[:, None]
    tmp36 = 64.0
    tmp37 = tmp35 / tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 4), (4, 1))
    assert_size_stride(arg1_1, (16, 4), (4, 1))
    assert_size_stride(arg2_1, (64, 4), (4, 1))
    assert_size_stride(arg3_1, (64, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [neg, add, log, mul, policy_error, add_1, total_error], Original ATen: [aten.neg, aten.add, aten.log, aten.mul, aten.sum, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_log_mean_mul_neg_sum_0.run(buf2, arg0_1, arg1_1, arg2_1, arg3_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
