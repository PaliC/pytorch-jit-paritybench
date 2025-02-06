# AOT ID: ['19_inference']
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


cpp_fused_lift_fresh_prod_0 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(4);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = tmp3 ? tmp4 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp8;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/ar/carkjyhtjtpbqmllly26eszb2g2o56z3mokryx3ojbqjfctucph2.py
# Topologically Sorted Source Nodes: [mul, std, mul_1, z, mul_2, wrapped_log, mul_3, log, pow_1, mul_4, add_1, sum_1, log_q], Original ATen: [aten.mul, aten.exp, aten.add, aten.log, aten.pow, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   log => log_1
#   log_q => sub
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   pow_1 => pow_1
#   std => exp
#   sum_1 => sum_1
#   wrapped_log => full_default
#   z => add
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %randn), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%prod, -0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.8378770664093453), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %full_default), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exp,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%randn, 2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, %mul_4), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_1, [2, 3, 4]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %sum_1), kwargs = {})
triton_per_fused_add_exp_log_mul_pow_sub_sum_1 = async_compile.triton('triton_per_fused_add_exp_log_mul_pow_sub_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_exp_log_mul_pow_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_exp_log_mul_pow_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (r1 + 32*x0), xmask, other=0.0)
    tmp19 = in_ptr2
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tl_math.log(tmp7)
    tmp12 = tmp8 * tmp8
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp19.to(tl.float32)
    tmp21 = -0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float64)
    tmp24 = tl.full([1, 1], 1.8378770664093453, tl.float64)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 - tmp18
    tl.store(out_ptr0 + (r1 + 32*x0), tmp10, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [eps], Original ATen: [aten.randn]
        buf0 = torch.ops.aten.randn.default([4, 1, 2, 4, 4], device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
    buf3 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_0(buf3)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((4, 1, 2, 4, 4), (32, 32, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf5 = reinterpret_tensor(buf4, (4, 1), (1, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [mul, std, mul_1, z, mul_2, wrapped_log, mul_3, log, pow_1, mul_4, add_1, sum_1, log_q], Original ATen: [aten.mul, aten.exp, aten.add, aten.log, aten.pow, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_exp_log_mul_pow_sub_sum_1.run(buf5, arg0_1, buf1, buf3.item(), buf2, 4, 32, grid=grid(4), stream=stream0)
        del arg0_1
        del buf1
        del buf3
    return (buf2, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
