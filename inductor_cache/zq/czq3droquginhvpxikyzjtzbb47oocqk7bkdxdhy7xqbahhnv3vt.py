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


# kernel path: inductor_cache/sn/csn4fpi6evmfkvbm64acofluvrvtkkrpn7tqbwqnwgogut2hwrot.py
# Topologically Sorted Source Nodes: [normalize, normalize_1, sub], Original ATen: [aten.linalg_vector_norm, aten.div, aten.sub]
# Source node to ATen node mapping:
#   normalize => div, pow_2, sum_1
#   normalize_1 => div_1, pow_5, sum_2
#   sub => sub
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view, %expand), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_1, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_1, %expand_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
triton_per_fused_div_linalg_vector_norm_sub_0 = async_compile.triton('triton_per_fused_div_linalg_vector_norm_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_linalg_vector_norm_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_linalg_vector_norm_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr0 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr0 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp18 = tl.load(in_ptr1 + (r1 + 64*x0), xmask, other=0.0)
    tmp20 = tl.load(in_ptr1 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp23 = tl.load(in_ptr1 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp18 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 / tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = libdevice.sqrt(tmp17)
    tmp36 = 1e-12
    tmp37 = triton_helpers.maximum(tmp35, tmp36)
    tmp38 = tmp12 / tmp37
    tmp39 = libdevice.sqrt(tmp34)
    tmp40 = triton_helpers.maximum(tmp39, tmp36)
    tmp41 = tmp29 / tmp40
    tmp42 = tmp38 - tmp41
    tl.store(out_ptr2 + (r1 + 16*x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/em/cemz5psckrx6aoefkbw532anyskziul7snl6pji3evqxtt2qtbds.py
# Topologically Sorted Source Nodes: [pow_3, mean_2], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   pow_3 => pow_7
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_7,), kwargs = {})
triton_per_fused_mean_pow_1 = async_compile.triton('triton_per_fused_mean_pow_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
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
        buf2 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [normalize, normalize_1, sub], Original ATen: [aten.linalg_vector_norm, aten.div, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_sub_0.run(arg0_1, arg1_1, buf2, 4, 16, grid=grid(4), stream=stream0)
        del arg0_1
        del arg1_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [pow_3, mean_2], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_1.run(buf4, buf2, 1, 64, grid=grid(1), stream=stream0)
        del buf2
    return (buf4, )


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
