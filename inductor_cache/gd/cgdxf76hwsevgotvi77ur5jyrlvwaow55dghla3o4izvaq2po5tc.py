# AOT ID: ['25_inference']
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


# kernel path: inductor_cache/i6/ci6gywcsmm6tyosu2rl27eourz5lrorkt66in5fgzrs7w2lykaw7.py
# Topologically Sorted Source Nodes: [max_1, a_tilde, mul, add, clamp, pow_1, za, pow_2, sub_1, a_tilde_1, mul_2, add_1, clamp_1, pow_3, za_1, pow_4, sub_2, a_tilde_2, mul_4, add_2, clamp_2, pow_5, za_2, truediv, pow_7, sub_4, truediv_1, neg, add_3, sm, mul_6, add_4, clamp_3, x, y_hat, pow_9, sub_6, truediv_2, neg_1, pow_10, sum_4, sub_7, truediv_3, out, truediv_4, sum_5], Original ATen: [aten.max, aten.sub, aten.mul, aten.add, aten.clamp, aten.pow, aten.sum, aten.reciprocal, aten.div, aten.neg, aten.index, aten.rsub]
# Source node to ATen node mapping:
#   a_tilde => sub
#   a_tilde_1 => mul_1
#   a_tilde_2 => mul_3
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   clamp => clamp_min
#   clamp_1 => clamp_min_1
#   clamp_2 => clamp_min_2
#   clamp_3 => clamp_min_3
#   max_1 => max_1
#   mul => mul
#   mul_2 => mul_2
#   mul_4 => mul_4
#   mul_6 => mul_7
#   neg => neg
#   neg_1 => neg_1
#   out => sub_8
#   pow_1 => pow_1
#   pow_10 => pow_10
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_7 => pow_7
#   pow_9 => pow_9
#   sm => sub_5
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_4 => sub_4
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sum_4 => sum_4
#   sum_5 => sum_5
#   truediv => mul_6, reciprocal
#   truediv_1 => div
#   truediv_2 => div_1
#   truediv_3 => div_2
#   truediv_4 => div_3
#   x => pow_8
#   y_hat => index
#   za => sum_1
#   za_1 => sum_2
#   za_2 => sum_3
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%arg0_1, 1, True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min, -0.3333333333333333), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, -3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %sub_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, -3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_1, -0.3333333333333333), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, -3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %sub_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, -3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_2, -0.3333333333333333), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sum_3,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_6, -3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_7, 1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_4, -3), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %getitem), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %add_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, -3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, 1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_3, -0.3333333333333333), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%pow_8, [%iota_default, %arg1_1]), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%index, -3), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_9, 1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_6, -3), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_8, -2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_10, [1]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sum_4), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_7, -2), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg_1, %div_2), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_8, 4), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%div_3,), kwargs = {})
triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0 = async_compile.triton('triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp127 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 - tmp6
    tmp8 = -3.0
    tmp9 = tmp7 * tmp8
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = -0.3333333333333333
    tmp15 = libdevice.pow(tmp13, tmp14)
    tmp16 = tmp1 - tmp6
    tmp17 = tmp16 * tmp8
    tmp18 = tmp17 + tmp10
    tmp19 = triton_helpers.maximum(tmp18, tmp12)
    tmp20 = libdevice.pow(tmp19, tmp14)
    tmp21 = tmp15 + tmp20
    tmp22 = tmp3 - tmp6
    tmp23 = tmp22 * tmp8
    tmp24 = tmp23 + tmp10
    tmp25 = triton_helpers.maximum(tmp24, tmp12)
    tmp26 = libdevice.pow(tmp25, tmp14)
    tmp27 = tmp21 + tmp26
    tmp28 = tmp5 - tmp6
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp10
    tmp31 = triton_helpers.maximum(tmp30, tmp12)
    tmp32 = libdevice.pow(tmp31, tmp14)
    tmp33 = tmp27 + tmp32
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 * tmp35
    tmp38 = tmp37 * tmp7
    tmp39 = tmp38 * tmp8
    tmp40 = tmp39 + tmp10
    tmp41 = triton_helpers.maximum(tmp40, tmp12)
    tmp42 = libdevice.pow(tmp41, tmp14)
    tmp43 = tmp37 * tmp16
    tmp44 = tmp43 * tmp8
    tmp45 = tmp44 + tmp10
    tmp46 = triton_helpers.maximum(tmp45, tmp12)
    tmp47 = libdevice.pow(tmp46, tmp14)
    tmp48 = tmp42 + tmp47
    tmp49 = tmp37 * tmp22
    tmp50 = tmp49 * tmp8
    tmp51 = tmp50 + tmp10
    tmp52 = triton_helpers.maximum(tmp51, tmp12)
    tmp53 = libdevice.pow(tmp52, tmp14)
    tmp54 = tmp48 + tmp53
    tmp55 = tmp37 * tmp28
    tmp56 = tmp55 * tmp8
    tmp57 = tmp56 + tmp10
    tmp58 = triton_helpers.maximum(tmp57, tmp12)
    tmp59 = libdevice.pow(tmp58, tmp14)
    tmp60 = tmp54 + tmp59
    tmp61 = tmp34 / tmp60
    tmp62 = tmp61 * tmp61
    tmp63 = tmp62 * tmp61
    tmp64 = tmp63 * tmp7
    tmp65 = tmp64 * tmp8
    tmp66 = tmp65 + tmp10
    tmp67 = triton_helpers.maximum(tmp66, tmp12)
    tmp68 = libdevice.pow(tmp67, tmp14)
    tmp69 = tmp63 * tmp16
    tmp70 = tmp69 * tmp8
    tmp71 = tmp70 + tmp10
    tmp72 = triton_helpers.maximum(tmp71, tmp12)
    tmp73 = libdevice.pow(tmp72, tmp14)
    tmp74 = tmp68 + tmp73
    tmp75 = tmp63 * tmp22
    tmp76 = tmp75 * tmp8
    tmp77 = tmp76 + tmp10
    tmp78 = triton_helpers.maximum(tmp77, tmp12)
    tmp79 = libdevice.pow(tmp78, tmp14)
    tmp80 = tmp74 + tmp79
    tmp81 = tmp63 * tmp28
    tmp82 = tmp81 * tmp8
    tmp83 = tmp82 + tmp10
    tmp84 = triton_helpers.maximum(tmp83, tmp12)
    tmp85 = libdevice.pow(tmp84, tmp14)
    tmp86 = tmp80 + tmp85
    tmp87 = tmp34 / tmp86
    tmp88 = tmp87 * tmp10
    tmp89 = tmp34 / tmp88
    tmp90 = tmp89 * tmp89
    tmp91 = tmp90 * tmp89
    tmp92 = tmp91 - tmp10
    tmp93 = tmp92 * tmp14
    tmp94 = -tmp93
    tmp95 = tmp94 + tmp6
    tmp96 = tmp0 - tmp95
    tmp97 = tmp96 * tmp8
    tmp98 = tmp97 + tmp10
    tmp99 = triton_helpers.maximum(tmp98, tmp12)
    tmp100 = libdevice.pow(tmp99, tmp14)
    tmp101 = tmp34 / tmp100
    tmp102 = tmp101 * tmp101
    tmp103 = tmp1 - tmp95
    tmp104 = tmp103 * tmp8
    tmp105 = tmp104 + tmp10
    tmp106 = triton_helpers.maximum(tmp105, tmp12)
    tmp107 = libdevice.pow(tmp106, tmp14)
    tmp108 = tmp34 / tmp107
    tmp109 = tmp108 * tmp108
    tmp110 = tmp102 + tmp109
    tmp111 = tmp3 - tmp95
    tmp112 = tmp111 * tmp8
    tmp113 = tmp112 + tmp10
    tmp114 = triton_helpers.maximum(tmp113, tmp12)
    tmp115 = libdevice.pow(tmp114, tmp14)
    tmp116 = tmp34 / tmp115
    tmp117 = tmp116 * tmp116
    tmp118 = tmp110 + tmp117
    tmp119 = tmp5 - tmp95
    tmp120 = tmp119 * tmp8
    tmp121 = tmp120 + tmp10
    tmp122 = triton_helpers.maximum(tmp121, tmp12)
    tmp123 = libdevice.pow(tmp122, tmp14)
    tmp124 = tmp34 / tmp123
    tmp125 = tmp124 * tmp124
    tmp126 = tmp118 + tmp125
    tmp128 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp129 = tmp127 + tmp128
    tmp130 = tmp127 < 0
    tmp131 = tl.where(tmp130, tmp129, tmp127)
    tl.device_assert((0 <= tmp131) & (tmp131 < 4), "index out of bounds: 0 <= tmp131 < 4")
    tmp133 = tl.load(in_ptr0 + (r0 + 16*tmp131 + 64*r1), None)
    tmp134 = tmp133 - tmp95
    tmp135 = tmp134 * tmp8
    tmp136 = tmp135 + tmp10
    tmp137 = triton_helpers.maximum(tmp136, tmp12)
    tmp138 = libdevice.pow(tmp137, tmp14)
    tmp139 = tmp34 / tmp138
    tmp140 = tmp139 * tmp139
    tmp141 = tmp140 * tmp139
    tmp142 = tmp141 - tmp10
    tmp143 = tmp142 * tmp14
    tmp144 = -tmp143
    tmp145 = tmp10 - tmp126
    tmp146 = -0.5
    tmp147 = tmp145 * tmp146
    tmp148 = tmp144 - tmp147
    tmp149 = 0.25
    tmp150 = tmp148 * tmp149
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp153 = tl.sum(tmp151, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp153, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf5 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, a_tilde, mul, add, clamp, pow_1, za, pow_2, sub_1, a_tilde_1, mul_2, add_1, clamp_1, pow_3, za_1, pow_4, sub_2, a_tilde_2, mul_4, add_2, clamp_2, pow_5, za_2, truediv, pow_7, sub_4, truediv_1, neg, add_3, sm, mul_6, add_4, clamp_3, x, y_hat, pow_9, sub_6, truediv_2, neg_1, pow_10, sum_4, sub_7, truediv_3, out, truediv_4, sum_5], Original ATen: [aten.max, aten.sub, aten.mul, aten.add, aten.clamp, aten.pow, aten.sum, aten.reciprocal, aten.div, aten.neg, aten.index, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0.run(arg0_1, arg1_1, buf5, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
