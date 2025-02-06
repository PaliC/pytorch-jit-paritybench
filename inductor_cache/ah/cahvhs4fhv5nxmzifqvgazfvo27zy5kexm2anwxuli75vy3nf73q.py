# AOT ID: ['3_inference']
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


# kernel path: inductor_cache/2x/c2xg3rrptq5vphggsjkl3xqyhbgnwuocgsrqmk4qgdz2tiz3px5q.py
# Topologically Sorted Source Nodes: [clamp, mul, exp, mul_1, x, pow_1, exp_1, add_1, sub, sub_1, sum_1], Original ATen: [aten.clamp, aten.mul, aten.exp, aten.add, aten.pow, aten.sub, aten.sum]
# Source node to ATen node mapping:
#   add_1 => add_1
#   clamp => clamp_max, clamp_min
#   exp => exp
#   exp_1 => exp_1
#   mul => mul
#   mul_1 => mul_1
#   pow_1 => pow_1
#   sub => sub
#   sub_1 => sub_1
#   sum_1 => sum_1
#   x => add
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%getitem_1, -30.0), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 20.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %randn), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %mul_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%getitem, 2), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%clamp_max,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %exp_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, 1.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %clamp_max), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub_1, [1, 2, 3]), kwargs = {})
triton_per_fused_add_clamp_exp_mul_pow_sub_sum_0 = async_compile.triton('triton_per_fused_add_clamp_exp_mul_pow_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_exp_mul_pow_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_exp_mul_pow_sub_sum_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp9 = tl.load(in_out_ptr0 + (r1 + 32*x0), xmask, other=0.0)
    tmp2 = -30.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 20.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp0 + tmp10
    tmp12 = tmp0 * tmp0
    tmp13 = tl_math.exp(tmp5)
    tmp14 = tmp12 + tmp13
    tmp15 = 1.0
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 - tmp5
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + 32*x0), tmp11, xmask)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5z/c5ztwl5dexkaihbyfr7wfjw5akcwqjiz33lfesjszqt7lobkctoi.py
# Topologically Sorted Source Nodes: [kl_loss, sum_2, kl_loss_1], Original ATen: [aten.mul, aten.sum, aten.div]
# Source node to ATen node mapping:
#   kl_loss => mul_2
#   kl_loss_1 => div
#   sum_2 => sum_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.5), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, 4), kwargs = {})
triton_poi_fused_div_mul_sum_1 = async_compile.triton('triton_poi_fused_div_mul_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (2))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = tmp3 + tmp6
    tmp10 = tmp9 * tmp2
    tmp11 = tmp7 + tmp10
    tmp14 = tmp13 * tmp2
    tmp15 = tmp11 + tmp14
    tmp16 = 0.25
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [randn], Original ATen: [aten.randn]
        buf0 = torch.ops.aten.randn.default([4, 2, 4, 4], device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [clamp, mul, exp, mul_1, x, pow_1, exp_1, add_1, sub, sub_1, sum_1], Original ATen: [aten.clamp, aten.mul, aten.exp, aten.add, aten.pow, aten.sub, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_exp_mul_pow_sub_sum_0.run(buf2, arg0_1, buf3, 4, 32, grid=grid(4), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [kl_loss, sum_2, kl_loss_1], Original ATen: [aten.mul, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mul_sum_1.run(buf3, buf4, 1, grid=grid(1), stream=stream0)
        del buf3
    return (buf2, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
