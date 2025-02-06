# AOT ID: ['11_inference']
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


# kernel path: inductor_cache/kx/ckx4neweb4a4g4qthiwbrwqbpvuxlytl7n3kjr3l74o3ixns37i4.py
# Topologically Sorted Source Nodes: [add_6, add_7, sub_11, pow_3, left, add_8, add_9, sub_12, pow_4, right, rho2, pow_1, pow_2, add_2, c2, truediv_5, w2, sub_10, h2, truediv_3, atan, w1, sub_8, h1, truediv_4, atan_1, sub_13, pow_5, v, pow_6, sub_1, sub_2, ap, sub_3, sub_4, ag, add, overlap, sub_5, union, ious, sub_14, add_11, truediv_6, add_12, cious, loss, loss_1, loss_2], Original ATen: [aten.add, aten.sub, aten.pow, aten.div, aten.atan, aten.mul, aten.rsub, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_11 => add_11
#   add_12 => add_12
#   add_2 => add_2
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   add_9 => add_9
#   ag => mul_2
#   ap => mul_1
#   atan => atan
#   atan_1 => atan_1
#   c2 => add_3
#   cious => sub_15
#   h1 => add_4
#   h2 => add_5
#   ious => div
#   left => div_1
#   loss => sub_16
#   loss_1 => mean
#   loss_2 => mul_4
#   overlap => mul
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   rho2 => add_10
#   right => div_2
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_8 => sub_8
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   union => add_1
#   v => mul_3
#   w1 => sub_7
#   w2 => sub_9
# Graph fragment:
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_16, %select_18), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_12, %select_14), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_7), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_11, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_3, 4), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_17, %select_19), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_13, %select_15), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_9), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_4, 4), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %div_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_10, 2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_11, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 1e-06), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_10, %add_3), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_18, %select_16), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_19, %select_17), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_10, 1e-06), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %add_5), kwargs = {})
#   %atan : [num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%div_3,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_14, %select_12), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_15, %select_13), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_8, 1e-06), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_7, %add_4), kwargs = {})
#   %atan_1 : [num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%div_4,), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%atan, %atan_1), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_13, 2), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_5, 0.4052847345693511), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_3, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %sub_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %select_1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_5, 1e-06), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %add_1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_14, %mul_3), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_6, %add_11), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_5, %div_6), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %add_12), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_15), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_16,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0 = async_compile.triton('triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = 0.25
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp17 * tmp8
    tmp19 = tmp9 + tmp18
    tmp20 = triton_helpers.maximum(tmp4, tmp1)
    tmp21 = triton_helpers.minimum(tmp3, tmp0)
    tmp22 = tmp20 - tmp21
    tmp23 = 0.0
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tmp24 * tmp24
    tmp26 = triton_helpers.maximum(tmp14, tmp11)
    tmp27 = triton_helpers.minimum(tmp13, tmp10)
    tmp28 = tmp26 - tmp27
    tmp29 = triton_helpers.maximum(tmp28, tmp23)
    tmp30 = tmp29 * tmp29
    tmp31 = tmp25 + tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tmp19 / tmp33
    tmp35 = tmp1 - tmp0
    tmp36 = tmp11 - tmp10
    tmp37 = tmp36 + tmp32
    tmp38 = tmp35 / tmp37
    tmp39 = libdevice.atan(tmp38)
    tmp40 = tmp4 - tmp3
    tmp41 = tmp14 - tmp13
    tmp42 = tmp41 + tmp32
    tmp43 = tmp40 / tmp42
    tmp44 = libdevice.atan(tmp43)
    tmp45 = tmp39 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = 0.4052847345693511
    tmp48 = tmp46 * tmp47
    tmp49 = triton_helpers.minimum(tmp4, tmp1)
    tmp50 = triton_helpers.maximum(tmp3, tmp0)
    tmp51 = tmp49 - tmp50
    tmp52 = triton_helpers.maximum(tmp51, tmp23)
    tmp53 = triton_helpers.minimum(tmp14, tmp11)
    tmp54 = triton_helpers.maximum(tmp13, tmp10)
    tmp55 = tmp53 - tmp54
    tmp56 = triton_helpers.maximum(tmp55, tmp23)
    tmp57 = tmp52 * tmp56
    tmp58 = tmp40 * tmp41
    tmp59 = tmp35 * tmp36
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 - tmp57
    tmp62 = tmp61 + tmp32
    tmp63 = tmp57 / tmp62
    tmp64 = tmp48 * tmp48
    tmp65 = 1.0
    tmp66 = tmp65 - tmp63
    tmp67 = tmp66 + tmp48
    tmp68 = tmp64 / tmp67
    tmp69 = tmp34 + tmp68
    tmp70 = tmp63 - tmp69
    tmp71 = tmp65 - tmp70
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, RBLOCK])
    tmp74 = tl.sum(tmp72, 1)[:, None]
    tmp75 = 64.0
    tmp76 = tmp74 / tmp75
    tmp77 = tmp76 * tmp65
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp77, None)
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
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [add_6, add_7, sub_11, pow_3, left, add_8, add_9, sub_12, pow_4, right, rho2, pow_1, pow_2, add_2, c2, truediv_5, w2, sub_10, h2, truediv_3, atan, w1, sub_8, h1, truediv_4, atan_1, sub_13, pow_5, v, pow_6, sub_1, sub_2, ap, sub_3, sub_4, ag, add, overlap, sub_5, union, ious, sub_14, add_11, truediv_6, add_12, cious, loss, loss_1, loss_2], Original ATen: [aten.add, aten.sub, aten.pow, aten.div, aten.atan, aten.mul, aten.rsub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0.run(buf5, arg1_1, arg0_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
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
