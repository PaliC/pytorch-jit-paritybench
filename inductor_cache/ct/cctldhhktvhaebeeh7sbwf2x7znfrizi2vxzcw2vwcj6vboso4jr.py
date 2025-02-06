# AOT ID: ['18_inference']
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


# kernel path: inductor_cache/ur/cur4fmtg4ivdf5pgip5klclyjcrtht6jzcfnfvkpjwzpgy2cx4nc.py
# Topologically Sorted Source Nodes: [sub_1, sub_2, ap, sub_3, sub_4, ag, add, overlap, sub_5, union, ious, add_4, add_5, sub_7, pow_3, left, add_6, add_7, sub_8, pow_4, right, rho2, pow_1, pow_2, add_2, c2, truediv_3, dious, loss, loss_1, loss_2], Original ATen: [aten.sub, aten.mul, aten.add, aten.div, aten.pow, aten.rsub, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   ag => mul_2
#   ap => mul_1
#   c2 => add_3
#   dious => sub_9
#   ious => div
#   left => div_1
#   loss => sub_10
#   loss_1 => mean
#   loss_2 => mul_3
#   overlap => mul
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   rho2 => add_8
#   right => div_2
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_7 => sub_7
#   sub_8 => sub_8
#   truediv_3 => div_3
#   union => add_1
# Graph fragment:
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
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %add_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_16, %select_18), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_12, %select_14), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %add_5), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_3, 4), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_17, %select_19), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_13, %select_15), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %add_7), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_4, 4), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %div_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_10, 2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_11, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 1e-06), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_8, %add_3), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_3), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_9), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_10,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
triton_per_fused_add_div_mean_mul_pow_rsub_sub_0 = async_compile.triton('triton_per_fused_add_div_mean_mul_pow_rsub_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_rsub_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp11 - tmp14
    tmp16 = triton_helpers.maximum(tmp15, tmp7)
    tmp17 = tmp8 * tmp16
    tmp18 = tmp0 - tmp3
    tmp19 = tmp9 - tmp12
    tmp20 = tmp18 * tmp19
    tmp21 = tmp1 - tmp4
    tmp22 = tmp10 - tmp13
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tmp24 - tmp17
    tmp26 = tmp4 + tmp1
    tmp27 = tmp3 + tmp0
    tmp28 = tmp26 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = 0.25
    tmp31 = tmp29 * tmp30
    tmp32 = tmp13 + tmp10
    tmp33 = tmp12 + tmp9
    tmp34 = tmp32 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tmp35 * tmp30
    tmp37 = tmp31 + tmp36
    tmp38 = triton_helpers.maximum(tmp0, tmp1)
    tmp39 = triton_helpers.minimum(tmp3, tmp4)
    tmp40 = tmp38 - tmp39
    tmp41 = triton_helpers.maximum(tmp40, tmp7)
    tmp42 = tmp41 * tmp41
    tmp43 = triton_helpers.maximum(tmp9, tmp10)
    tmp44 = triton_helpers.minimum(tmp12, tmp13)
    tmp45 = tmp43 - tmp44
    tmp46 = triton_helpers.maximum(tmp45, tmp7)
    tmp47 = tmp46 * tmp46
    tmp48 = tmp42 + tmp47
    tmp49 = 1e-06
    tmp50 = tmp48 + tmp49
    tmp51 = tmp37 / tmp50
    tmp52 = tmp25 + tmp49
    tmp53 = tmp17 / tmp52
    tmp54 = tmp53 - tmp51
    tmp55 = 1.0
    tmp56 = tmp55 - tmp54
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp60 = 64.0
    tmp61 = tmp59 / tmp60
    tmp62 = tmp61 * tmp55
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp62, None)
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
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [sub_1, sub_2, ap, sub_3, sub_4, ag, add, overlap, sub_5, union, ious, add_4, add_5, sub_7, pow_3, left, add_6, add_7, sub_8, pow_4, right, rho2, pow_1, pow_2, add_2, c2, truediv_3, dious, loss, loss_1, loss_2], Original ATen: [aten.sub, aten.mul, aten.add, aten.div, aten.pow, aten.rsub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_pow_rsub_sub_0.run(buf4, arg0_1, arg1_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
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
