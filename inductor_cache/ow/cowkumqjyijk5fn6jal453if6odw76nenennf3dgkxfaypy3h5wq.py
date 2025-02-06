# AOT ID: ['1_inference']
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


# kernel path: inductor_cache/c7/cc7ksp4yhbvdm4p4mndjlbhmwxkzcq3oaw4wswxdu676e3bxxrfz.py
# Topologically Sorted Source Nodes: [smooth_l1_loss, sum_1, loss_transl, mul, smooth_l1_loss_1, sum_2, loss_rot, mul_1, total_loss], Original ATen: [aten.smooth_l1_loss, aten.sum, aten.mean, aten.mul, aten.add]
# Source node to ATen node mapping:
#   loss_rot => mean_1
#   loss_transl => mean
#   mul => mul_2
#   mul_1 => mul_3
#   smooth_l1_loss => abs_1, div, lt, mul, pow_1, sub, sub_1, where
#   smooth_l1_loss_1 => abs_2, div_1, lt_1, mul_1, pow_2, sub_2, sub_3, where_1
#   sum_1 => sum_1
#   sum_2 => sum_2
#   total_loss => add
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg0_1), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 1.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %div, %sub_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %arg2_1), kwargs = {})
#   %abs_2 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_2, 1.0), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_2, 2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_2, 0.5), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %div_1, %sub_3), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_1, [1]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
triton_per_fused_add_mean_mul_smooth_l1_loss_sum_0 = async_compile.triton('triton_per_fused_add_mean_mul_smooth_l1_loss_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_smooth_l1_loss_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_mul_smooth_l1_loss_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp23 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp24 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp34 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp35 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp48 = tl.load(in_ptr2 + (r0 + 64*r1), None)
    tmp49 = tl.load(in_ptr3 + (r0 + 64*r1), None)
    tmp58 = tl.load(in_ptr2 + (16 + r0 + 64*r1), None)
    tmp59 = tl.load(in_ptr3 + (16 + r0 + 64*r1), None)
    tmp69 = tl.load(in_ptr2 + (32 + r0 + 64*r1), None)
    tmp70 = tl.load(in_ptr3 + (32 + r0 + 64*r1), None)
    tmp80 = tl.load(in_ptr2 + (48 + r0 + 64*r1), None)
    tmp81 = tl.load(in_ptr3 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 < tmp4
    tmp6 = tmp3 * tmp3
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp3 - tmp7
    tmp11 = tl.where(tmp5, tmp9, tmp10)
    tmp14 = tmp12 - tmp13
    tmp15 = tl_math.abs(tmp14)
    tmp16 = tmp15 < tmp4
    tmp17 = tmp15 * tmp15
    tmp18 = tmp17 * tmp7
    tmp19 = tmp18 * tmp4
    tmp20 = tmp15 - tmp7
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp11 + tmp21
    tmp25 = tmp23 - tmp24
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp26 < tmp4
    tmp28 = tmp26 * tmp26
    tmp29 = tmp28 * tmp7
    tmp30 = tmp29 * tmp4
    tmp31 = tmp26 - tmp7
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tmp22 + tmp32
    tmp36 = tmp34 - tmp35
    tmp37 = tl_math.abs(tmp36)
    tmp38 = tmp37 < tmp4
    tmp39 = tmp37 * tmp37
    tmp40 = tmp39 * tmp7
    tmp41 = tmp40 * tmp4
    tmp42 = tmp37 - tmp7
    tmp43 = tl.where(tmp38, tmp41, tmp42)
    tmp44 = tmp33 + tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None]
    tmp50 = tmp48 - tmp49
    tmp51 = tl_math.abs(tmp50)
    tmp52 = tmp51 < tmp4
    tmp53 = tmp51 * tmp51
    tmp54 = tmp53 * tmp7
    tmp55 = tmp54 * tmp4
    tmp56 = tmp51 - tmp7
    tmp57 = tl.where(tmp52, tmp55, tmp56)
    tmp60 = tmp58 - tmp59
    tmp61 = tl_math.abs(tmp60)
    tmp62 = tmp61 < tmp4
    tmp63 = tmp61 * tmp61
    tmp64 = tmp63 * tmp7
    tmp65 = tmp64 * tmp4
    tmp66 = tmp61 - tmp7
    tmp67 = tl.where(tmp62, tmp65, tmp66)
    tmp68 = tmp57 + tmp67
    tmp71 = tmp69 - tmp70
    tmp72 = tl_math.abs(tmp71)
    tmp73 = tmp72 < tmp4
    tmp74 = tmp72 * tmp72
    tmp75 = tmp74 * tmp7
    tmp76 = tmp75 * tmp4
    tmp77 = tmp72 - tmp7
    tmp78 = tl.where(tmp73, tmp76, tmp77)
    tmp79 = tmp68 + tmp78
    tmp82 = tmp80 - tmp81
    tmp83 = tl_math.abs(tmp82)
    tmp84 = tmp83 < tmp4
    tmp85 = tmp83 * tmp83
    tmp86 = tmp85 * tmp7
    tmp87 = tmp86 * tmp4
    tmp88 = tmp83 - tmp7
    tmp89 = tl.where(tmp84, tmp87, tmp88)
    tmp90 = tmp79 + tmp89
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
    tmp93 = tl.sum(tmp91, 1)[:, None]
    tmp94 = 64.0
    tmp95 = tmp47 / tmp94
    tmp96 = tmp95 * tmp4
    tmp97 = tmp93 / tmp94
    tmp98 = tmp97 * tmp4
    tmp99 = tmp96 + tmp98
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp99, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg3_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [smooth_l1_loss, sum_1, loss_transl, mul, smooth_l1_loss_1, sum_2, loss_rot, mul_1, total_loss], Original ATen: [aten.smooth_l1_loss, aten.sum, aten.mean, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_smooth_l1_loss_sum_0.run(buf4, arg1_1, arg0_1, arg3_1, arg2_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
