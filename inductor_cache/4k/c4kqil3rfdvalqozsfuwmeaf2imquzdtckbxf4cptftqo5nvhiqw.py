# AOT ID: ['71_inference']
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


# kernel path: inductor_cache/ye/cyetyxe2q6ii3xfaduzkxdcdvaqzd6m65yxjignd3tnpt2kaytnc.py
# Topologically Sorted Source Nodes: [alpha, setitem, eq_1, probs, sub, pt, sub_1, pow_1, mul, ce_loss, loss, loss_1], Original ATen: [aten.fill, aten.lift_fresh, aten.index_put, aten.eq, aten.sigmoid, aten.rsub, aten.where, aten.pow, aten.mul, aten.binary_cross_entropy_with_logits, aten.mean]
# Source node to ATen node mapping:
#   alpha => full_default
#   ce_loss => abs_1, exp, full_default_2, log1p, minimum, mul, neg, sub_1, sub_2, sub_3
#   eq_1 => eq_1
#   loss => mul_2
#   loss_1 => mean
#   mul => mul_1
#   pow_1 => pow_1
#   probs => sigmoid
#   pt => where
#   setitem => full_default_1, index_put
#   sub => sub
#   sub_1 => sub_4
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0.75), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.25), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%eq], %full_default_1), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg0_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %sigmoid, %sub), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %where), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_put, %pow_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg1_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %arg0_1), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_2, %arg0_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %sub_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sub_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_2,), kwargs = {})
triton_per_fused_binary_cross_entropy_with_logits_eq_fill_index_put_lift_fresh_mean_mul_pow_rsub_sigmoid_where_0 = async_compile.triton('triton_per_fused_binary_cross_entropy_with_logits_eq_fill_index_put_lift_fresh_mean_mul_pow_rsub_sigmoid_where_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_binary_cross_entropy_with_logits_eq_fill_index_put_lift_fresh_mean_mul_pow_rsub_sigmoid_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_binary_cross_entropy_with_logits_eq_fill_index_put_lift_fresh_mean_mul_pow_rsub_sigmoid_where_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp6 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 1.0
    tmp2 = tmp0 == tmp1
    tmp3 = 0.25
    tmp4 = 0.75
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp1 - tmp7
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp1 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp0
    tmp14 = tmp13 * tmp6
    tmp15 = 0.0
    tmp16 = triton_helpers.minimum(tmp15, tmp6)
    tmp17 = tl_math.abs(tmp6)
    tmp18 = -tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tmp16 - tmp20
    tmp22 = tmp14 - tmp21
    tmp23 = tmp12 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = 256.0
    tmp28 = tmp26 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp28, None)
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
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [alpha, setitem, eq_1, probs, sub, pt, sub_1, pow_1, mul, ce_loss, loss, loss_1], Original ATen: [aten.fill, aten.lift_fresh, aten.index_put, aten.eq, aten.sigmoid, aten.rsub, aten.where, aten.pow, aten.mul, aten.binary_cross_entropy_with_logits, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_binary_cross_entropy_with_logits_eq_fill_index_put_lift_fresh_mean_mul_pow_rsub_sigmoid_where_0.run(buf2, arg1_1, arg0_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf2, )


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
