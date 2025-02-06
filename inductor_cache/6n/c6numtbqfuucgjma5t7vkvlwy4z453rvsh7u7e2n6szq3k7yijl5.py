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


# kernel path: inductor_cache/2f/c2f5o3esow63pkclydgwh7uq5flgsdvl32t4sckl5n7croiwf63l.py
# Topologically Sorted Source Nodes: [mask_x, sub_1, grad_x, grad_x_1, sum_2, mask_y, sub_2, grad_y, grad_y_1, sum_3, image_loss], Original ATen: [aten.mul, aten.sub, aten.abs, aten.sum, aten.add]
# Source node to ATen node mapping:
#   grad_x => abs_1
#   grad_x_1 => mul_2
#   grad_y => abs_2
#   grad_y_1 => mul_4
#   image_loss => add
#   mask_x => mul_1
#   mask_y => mul_3
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_2 => sum_2
#   sum_3 => sum_3
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_9, %slice_12), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_6), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %abs_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1, 2]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_20, %slice_23), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_14, %slice_17), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %abs_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1, 2]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, %sum_3), kwargs = {})
triton_per_fused_abs_add_mul_sub_sum_0 = async_compile.triton('triton_per_fused_abs_add_mul_sub_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_mul_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = (rindex % 3)
    r3 = rindex // 3
    x0 = (xindex % 4)
    x1 = xindex // 4
    x5 = xindex
    r4 = rindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr0 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr2 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr1 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp6 - tmp10
    tmp12 = tl_math.abs(tmp11)
    tmp13 = tmp2 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 * tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp27 = tmp25 - tmp26
    tmp28 = tmp19 * tmp27
    tmp29 = tmp24 - tmp28
    tmp30 = tl_math.abs(tmp29)
    tmp31 = tmp20 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = tmp17 + tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tf/ctf7eqxo5lp3q7bn3ziq7iavukceftcif7k5gna4e562t2fc4flw.py
# Topologically Sorted Source Nodes: [M], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   M => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg0_1, [1, 2]), kwargs = {})
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r2 + 64*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
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
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mask_x, sub_1, grad_x, grad_x_1, sum_2, mask_y, sub_2, grad_y, grad_y_1, sum_3, image_loss], Original ATen: [aten.mul, aten.sub, aten.abs, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_mul_sub_sum_0.run(buf2, arg0_1, arg1_1, arg2_1, 16, 12, grid=grid(16), stream=stream0)
        del arg1_1
        del arg2_1
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [M], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(arg0_1, buf3, 16, 16, grid=grid(16), stream=stream0)
        del arg0_1
    return (buf2, buf3, )


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
