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
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/3v/c3vf6cs3br73co4wyer3dxvey4xcms44auttzkvh2ks4ypgto3nb.py
# Topologically Sorted Source Nodes: [l1_loss], Original ATen: [aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   l1_loss => abs_1, mean, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
triton_per_fused_abs_mean_sub_0 = async_compile.triton('triton_per_fused_abs_mean_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_mean_sub_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcsybfcqx5dap4lyhjieep6vvioxrl4dkchd2cr4r5tt35552h6.py
# Topologically Sorted Source Nodes: [pred_1, gt_1, l1_loss_1], Original ATen: [aten.avg_pool2d, aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   gt_1 => avg_pool2d_1
#   l1_loss_1 => abs_2, mean_1, sub_1
#   pred_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg0_1, [2, 2], [2, 2], [0, 0], False, False), kwargs = {})
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg1_1, [2, 2], [2, 2], [0, 0], False, False), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %avg_pool2d_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_2,), kwargs = {})
triton_per_fused_abs_avg_pool2d_mean_sub_1 = async_compile.triton('triton_per_fused_abs_avg_pool2d_mean_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_avg_pool2d_mean_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_avg_pool2d_mean_sub_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 2)
    r1 = rindex // 2
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (4 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (5 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp16 = tmp15 * tmp7
    tmp17 = tmp8 - tmp16
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp8, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp16, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/hc/chc6egha5dru37kbrxwi2357xfe2ezr47qi66eammieqjctw4oln.py
# Topologically Sorted Source Nodes: [l1_loss, mul, loss, l1_loss_1, mul_1, loss_1, pred_2, gt_2, l1_loss_2, mul_2, loss_2], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul, aten.add, aten.avg_pool2d]
# Source node to ATen node mapping:
#   gt_2 => avg_pool2d_3
#   l1_loss => abs_1, mean, sub
#   l1_loss_1 => abs_2, mean_1, sub_1
#   l1_loss_2 => abs_3, mean_2, sub_2
#   loss => add
#   loss_1 => add_1
#   loss_2 => add_2
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   pred_2 => avg_pool2d_2
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %avg_pool2d_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_1), kwargs = {})
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d, [2, 2], [2, 2], [0, 0], False, False), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d_1, [2, 2], [2, 2], [0, 0], False, False), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_2, %avg_pool2d_3), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 0.25), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_2), kwargs = {})
triton_per_fused_abs_add_avg_pool2d_mean_mul_sub_2 = async_compile.triton('triton_per_fused_abs_add_avg_pool2d_mean_mul_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_avg_pool2d_mean_mul_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_avg_pool2d_mean_mul_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr0 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, 1])
    tmp30 = tl.load(in_ptr2 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, 1])
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp16 = tmp15 * tmp7
    tmp17 = tmp8 - tmp16
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None]
    tmp24 = 256.0
    tmp25 = tmp23 / tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = 0.0
    tmp29 = tmp27 + tmp28
    tmp32 = 64.0
    tmp33 = tmp31 / tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp29 + tmp35
    tmp37 = 16.0
    tmp38 = tmp21 / tmp37
    tmp39 = tmp38 * tmp7
    tmp40 = tmp36 + tmp39
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp40, None)
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
        buf0 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [l1_loss], Original ATen: [aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_mean_sub_0.run(arg0_1, arg1_1, buf0, 1, 256, grid=grid(1), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [pred_1, gt_1, l1_loss_1], Original ATen: [aten.avg_pool2d, aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_avg_pool2d_mean_sub_1.run(arg0_1, arg1_1, buf1, buf2, buf3, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [l1_loss, mul, loss, l1_loss_1, mul_1, loss_1, pred_2, gt_2, l1_loss_2, mul_2, loss_2], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul, aten.add, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_avg_pool2d_mean_mul_sub_2.run(buf5, buf1, buf2, buf3, 1, 16, grid=grid(1), stream=stream0)
        del buf1
        del buf2
        del buf3
    return (buf5, )


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
