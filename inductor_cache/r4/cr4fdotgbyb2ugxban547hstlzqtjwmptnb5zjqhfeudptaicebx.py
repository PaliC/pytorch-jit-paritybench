# AOT ID: ['17_inference']
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


# kernel path: inductor_cache/4f/c4ftlljvkwiqrmyxjj2ykkn5y2byqqgudgebafubimfuvtjhdwsm.py
# Topologically Sorted Source Nodes: [prob, sub_2, pos_weight, mul_2, clamp, abs_1, neg, exp, add, log, sub, clamp_1, mul, pos_log_sig, mul_3, sub_3, neg_weight, mul_4, neg_1, clamp_2, abs_2, neg_2, exp_1, add_2, log_1, sub_1, clamp_3, mul_1, neg_log_sig, mul_5, add_4, loss, mul_6, sub_4, mul_7, avg_weight, mean, loss_1, mean_1], Original ATen: [aten.sigmoid, aten.rsub, aten.pow, aten.mul, aten.clamp, aten.abs, aten.neg, aten.exp, aten.add, aten.log, aten.sub, aten.mean, aten.div]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   abs_2 => abs_2
#   add => add
#   add_2 => add_2
#   add_4 => add_4
#   avg_weight => add_5
#   clamp => clamp_max
#   clamp_1 => clamp_max_1, clamp_min
#   clamp_2 => clamp_max_2
#   clamp_3 => clamp_max_3, clamp_min_1
#   exp => exp
#   exp_1 => exp_1
#   log => log
#   log_1 => log_1
#   loss => neg_3
#   loss_1 => div
#   mean => mean
#   mean_1 => mean_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   neg => neg
#   neg_1 => neg_1
#   neg_2 => neg_2
#   neg_log_sig => add_3
#   neg_weight => pow_2
#   pos_log_sig => add_1
#   pos_weight => pow_1
#   prob => sigmoid
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg0_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %pow_1), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%arg0_1, 0), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %log), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %mul), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg1_1), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sigmoid, 2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %pow_2), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg0_1,), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%neg_1, 0), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max_2, %log_1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_3, 0.5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, %mul_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_5), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_4,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %pow_1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg1_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %pow_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_5,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg_3, %mean), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%div,), kwargs = {})
triton_per_fused_abs_add_clamp_div_exp_log_mean_mul_neg_pow_rsub_sigmoid_sub_0 = async_compile.triton('triton_per_fused_abs_add_clamp_div_exp_log_mean_mul_neg_pow_rsub_sigmoid_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_clamp_div_exp_log_mean_mul_neg_pow_rsub_sigmoid_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_clamp_div_exp_log_mean_mul_neg_pow_rsub_sigmoid_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp4 * tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.minimum(tmp1, tmp7)
    tmp9 = tl_math.abs(tmp1)
    tmp10 = -tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tmp11 + tmp3
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp8 - tmp13
    tmp15 = triton_helpers.maximum(tmp1, tmp7)
    tmp16 = triton_helpers.minimum(tmp15, tmp7)
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tmp6 * tmp19
    tmp21 = tmp3 - tmp0
    tmp22 = tmp2 * tmp2
    tmp23 = tmp21 * tmp22
    tmp24 = -tmp1
    tmp25 = triton_helpers.minimum(tmp24, tmp7)
    tmp26 = tmp25 - tmp13
    tmp27 = tmp26 + tmp18
    tmp28 = tmp23 * tmp27
    tmp29 = tmp20 + tmp28
    tmp30 = -tmp29
    tmp31 = tmp6 + tmp23
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp35 = 256.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp30 / tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = tmp40 / tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp41, None)
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
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [prob, sub_2, pos_weight, mul_2, clamp, abs_1, neg, exp, add, log, sub, clamp_1, mul, pos_log_sig, mul_3, sub_3, neg_weight, mul_4, neg_1, clamp_2, abs_2, neg_2, exp_1, add_2, log_1, sub_1, clamp_3, mul_1, neg_log_sig, mul_5, add_4, loss, mul_6, sub_4, mul_7, avg_weight, mean, loss_1, mean_1], Original ATen: [aten.sigmoid, aten.rsub, aten.pow, aten.mul, aten.clamp, aten.abs, aten.neg, aten.exp, aten.add, aten.log, aten.sub, aten.mean, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_clamp_div_exp_log_mean_mul_neg_pow_rsub_sigmoid_sub_0.run(buf3, arg1_1, arg0_1, 1, 256, grid=grid(1), stream=stream0)
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
