# AOT ID: ['6_inference']
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

empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/ee/ceeru7dkpbvxsbpjzllvia355hilp43kxyo4gugyo3n7oylrres4.py
# Topologically Sorted Source Nodes: [add_, div_, noise, add__1, clamp_, round_, mul_, sub__1], Original ATen: [aten.add, aten.div, aten.sub, aten.clamp, aten.round, aten.mul]
# Source node to ATen node mapping:
#   add_ => add
#   add__1 => add_1
#   clamp_ => clamp_max, clamp_min
#   div_ => div
#   mul_ => mul
#   noise => sub
#   round_ => round_1
#   sub__1 => sub_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%rand, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %sub), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %round_1 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%clamp_max,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%round_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 1), kwargs = {})
#   %copy_ : [num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %sub_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_mul_round_sub_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5 - tmp3
    tmp7 = tmp4 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = triton_helpers.minimum(tmp9, tmp1)
    tmp11 = libdevice.nearbyint(tmp10)
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 - tmp1
    tl.store(out_ptr0 + (x0), tmp14, xmask)







def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [rand_like], Original ATen: [aten.rand_like]
        buf0 = torch.ops.aten.rand.default([4, 4, 4, 4], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        buf9 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [add_, div_, noise, add__1, clamp_, round_, mul_, sub__1], Original ATen: [aten.add, aten.div, aten.sub, aten.clamp, aten.round, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_round_sub_0[grid(256)](buf9, arg0_1, arg0_1, 256, XBLOCK=128, num_warps=4, num_stages=1)
        del buf9
    return (arg0_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
