# AOT ID: ['8_inference']
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


# kernel path: inductor_cache/3t/c3td4mcnu66vee67ezvlcvhtxifu4dlicrg6r2sr3jbtmxojmnrm.py
# Topologically Sorted Source Nodes: [eq_1, probs, sub_1, probs_1_gamma, ge, softplus, softplus_1, sub, log_probs, term1, probs_gamma, ge_1, neg, softplus_2, add, softplus_3, neg_1, log_1_probs, term2, where_2, coeff, setitem, mul_, loss, loss_1], Original ATen: [aten.eq, aten.sigmoid, aten.rsub, aten.pow, aten.ge, aten.softplus, aten.sub, aten.where, aten.mul, aten.neg, aten.add, aten.fill, aten.lift_fresh, aten.index_put, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   coeff => full_default
#   eq_1 => eq_1
#   ge => ge
#   ge_1 => ge_1
#   log_1_probs => where_5
#   log_probs => where_2
#   loss => neg_2
#   loss_1 => mean
#   mul_ => mul_6
#   neg => neg
#   neg_1 => neg_1
#   probs => sigmoid
#   probs_1_gamma => pow_2
#   probs_gamma => pow_1
#   setitem => full_default_1, index_put
#   softplus => div, exp, gt, log1p, mul, where
#   softplus_1 => div_1, exp_1, gt_1, log1p_1, mul_1, where_1
#   softplus_2 => div_2, exp_2, gt_2, log1p_2, mul_2, where_3
#   softplus_3 => div_3, exp_3, gt_3, log1p_3, mul_3, where_4
#   sub => sub
#   sub_1 => sub_1
#   term1 => mul_4
#   term2 => mul_5
#   where_2 => where_6
# Graph fragment:
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg0_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, -1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul, 50), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, -1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %arg0_1, %div), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_1, 50), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_1, 1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %arg0_1, %div_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %where_1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %where, %sub), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %where_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sigmoid, 2), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg0_1,), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, -1), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_2, 50), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_2, -1), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %arg0_1, %div_2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %where_3), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_3, 50), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_3,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_3,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_3, 1), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %arg0_1, %div_3), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_4,), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %add, %neg_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, %where_5), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %mul_4, %mul_5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0.75), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.25), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%eq], %full_default_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_6, %index_put), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_6,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%neg_2,), kwargs = {})
triton_per_fused_add_eq_fill_ge_index_put_lift_fresh_mean_mul_neg_pow_rsub_sigmoid_softplus_sub_where_0 = async_compile.triton('triton_per_fused_add_eq_fill_ge_index_put_lift_fresh_mean_mul_neg_pow_rsub_sigmoid_softplus_sub_where_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_eq_fill_ge_index_put_lift_fresh_mean_mul_neg_pow_rsub_sigmoid_softplus_sub_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_eq_fill_ge_index_put_lift_fresh_mean_mul_neg_pow_rsub_sigmoid_softplus_sub_where_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 1.0
    tmp2 = tmp0 == tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp1 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = 0.0
    tmp8 = tmp3 >= tmp7
    tmp9 = -1.0
    tmp10 = tmp3 * tmp9
    tmp11 = 50.0
    tmp12 = tmp10 > tmp11
    tmp13 = tl_math.exp(tmp10)
    tmp14 = libdevice.log1p(tmp13)
    tmp15 = tmp14 * tmp9
    tmp16 = tl.where(tmp12, tmp3, tmp15)
    tmp17 = tmp3 * tmp1
    tmp18 = tmp17 > tmp11
    tmp19 = tl_math.exp(tmp17)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tmp20 * tmp1
    tmp22 = tl.where(tmp18, tmp3, tmp21)
    tmp23 = tmp3 - tmp22
    tmp24 = tl.where(tmp8, tmp16, tmp23)
    tmp25 = tmp6 * tmp24
    tmp26 = tmp4 * tmp4
    tmp27 = -tmp3
    tmp28 = tmp27 + tmp16
    tmp29 = -tmp22
    tmp30 = tl.where(tmp8, tmp28, tmp29)
    tmp31 = tmp26 * tmp30
    tmp32 = tl.where(tmp2, tmp25, tmp31)
    tmp33 = 0.25
    tmp34 = 0.75
    tmp35 = tl.where(tmp2, tmp33, tmp34)
    tmp36 = tmp32 * tmp35
    tmp37 = -tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = 256.0
    tmp42 = tmp40 / tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp42, None)
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
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [eq_1, probs, sub_1, probs_1_gamma, ge, softplus, softplus_1, sub, log_probs, term1, probs_gamma, ge_1, neg, softplus_2, add, softplus_3, neg_1, log_1_probs, term2, where_2, coeff, setitem, mul_, loss, loss_1], Original ATen: [aten.eq, aten.sigmoid, aten.rsub, aten.pow, aten.ge, aten.softplus, aten.sub, aten.where, aten.mul, aten.neg, aten.add, aten.fill, aten.lift_fresh, aten.index_put, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_eq_fill_ge_index_put_lift_fresh_mean_mul_neg_pow_rsub_sigmoid_softplus_sub_where_0.run(buf3, arg1_1, arg0_1, 1, 256, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf3, )


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
