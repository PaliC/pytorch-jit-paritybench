# AOT ID: ['31_inference']
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


# kernel path: inductor_cache/e4/ce4rtmm56tvlajdzwtkfvnbmrqdvfx42uemq7pdrz5hp45kc52da.py
# Topologically Sorted Source Nodes: [sum_2, sum_3, add, mul, Iand1, Ior1, IoU1, sub_1, IoU, sum_5, sum_6, add_2, mul_1, Iand1_1, Ior1_1, IoU1_1, sub_3, IoU_1, sum_8, sum_9, add_4, mul_2, Iand1_2, Ior1_2, IoU1_2, sub_5, IoU_2, sum_11, sum_12, add_6, mul_3, Iand1_3, Ior1_3, IoU1_3, sub_7, IoU_3, truediv_4], Original ATen: [aten.sum, aten.add, aten.mul, aten.sub, aten.div, aten.rsub]
# Source node to ATen node mapping:
#   Iand1 => sum_1
#   Iand1_1 => sum_4
#   Iand1_2 => sum_7
#   Iand1_3 => sum_10
#   IoU => add_1
#   IoU1 => div
#   IoU1_1 => div_1
#   IoU1_2 => div_2
#   IoU1_3 => div_3
#   IoU_1 => add_3
#   IoU_2 => add_5
#   IoU_3 => add_7
#   Ior1 => sub
#   Ior1_1 => sub_2
#   Ior1_2 => sub_4
#   Ior1_3 => sub_6
#   add => add
#   add_2 => add_2
#   add_4 => add_4
#   add_6 => add_6
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   sub_1 => sub_1
#   sub_3 => sub_3
#   sub_5 => sub_5
#   sub_7 => sub_7
#   sum_11 => sum_11
#   sum_12 => sum_12
#   sum_2 => sum_2
#   sum_3 => sum_3
#   sum_5 => sum_5
#   sum_6 => sum_6
#   sum_8 => sum_8
#   sum_9 => sum_9
#   truediv_4 => div_4
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_2,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_3,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, %sum_3), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %select_1), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %sum_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sub), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, 0.0), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_6,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_7,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_5, %sum_6), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %sum_4 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %sum_4), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, %sub_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %sub_3), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_10,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_11,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_8, %sum_9), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %sum_7 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %sum_7), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_7, %sub_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %sub_5), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_14,), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%select_15,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_11, %sum_12), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %select_13), kwargs = {})
#   %sum_10 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_3,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %sum_10), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_10, %sub_6), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div_3), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %sub_7), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_7, 4), kwargs = {})
triton_per_fused_add_div_mul_rsub_sub_sum_0 = async_compile.triton('triton_per_fused_add_div_mul_rsub_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_rsub_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 12, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_rsub_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp12 = tl.load(in_ptr0 + (64 + r0), None)
    tmp16 = tl.load(in_ptr1 + (64 + r0), None)
    tmp24 = tl.load(in_ptr0 + (128 + r0), None)
    tmp28 = tl.load(in_ptr1 + (128 + r0), None)
    tmp36 = tl.load(in_ptr0 + (192 + r0), None)
    tmp40 = tl.load(in_ptr1 + (192 + r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp0 * tmp4
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp12 * tmp16
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = tmp24 * tmp28
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.sum(tmp33, 1)[:, None]
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None]
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.sum(tmp41, 1)[:, None]
    tmp44 = tmp36 * tmp40
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None]
    tmp48 = tmp3 + tmp7
    tmp49 = tmp48 - tmp11
    tmp50 = tmp11 / tmp49
    tmp51 = 1.0
    tmp52 = tmp51 - tmp50
    tmp53 = 0.0
    tmp54 = tmp52 + tmp53
    tmp55 = tmp15 + tmp19
    tmp56 = tmp55 - tmp23
    tmp57 = tmp23 / tmp56
    tmp58 = tmp51 - tmp57
    tmp59 = tmp54 + tmp58
    tmp60 = tmp27 + tmp31
    tmp61 = tmp60 - tmp35
    tmp62 = tmp35 / tmp61
    tmp63 = tmp51 - tmp62
    tmp64 = tmp59 + tmp63
    tmp65 = tmp39 + tmp43
    tmp66 = tmp65 - tmp47
    tmp67 = tmp47 / tmp66
    tmp68 = tmp51 - tmp67
    tmp69 = tmp64 + tmp68
    tmp70 = 0.25
    tmp71 = tmp69 * tmp70
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp71, None)
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
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf12 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [sum_2, sum_3, add, mul, Iand1, Ior1, IoU1, sub_1, IoU, sum_5, sum_6, add_2, mul_1, Iand1_1, Ior1_1, IoU1_1, sub_3, IoU_1, sum_8, sum_9, add_4, mul_2, Iand1_2, Ior1_2, IoU1_2, sub_5, IoU_2, sum_11, sum_12, add_6, mul_3, Iand1_3, Ior1_3, IoU1_3, sub_7, IoU_3, truediv_4], Original ATen: [aten.sum, aten.add, aten.mul, aten.sub, aten.div, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_rsub_sub_sum_0.run(buf12, arg1_1, arg0_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf12, )


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
