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


# kernel path: inductor_cache/3l/c3lobsl26bqvyv2oe75elkjwo26ogtmjwg22xr6hbfsuur4uqf5k.py
# Topologically Sorted Source Nodes: [output0], Original ATen: [aten.log_sigmoid_forward, aten.mul, aten.rsub, aten.neg, aten.add, aten.sum, aten.div, aten.mean]
# Source node to ATen node mapping:
#   output0 => abs_1, abs_2, add, div, exp, exp_1, full_default, full_default_1, log1p, log1p_1, mean, minimum, minimum_1, mul, mul_1, neg, neg_1, neg_2, neg_3, sub, sub_1, sub_2, sum_1
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %arg1_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg1_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %sub), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_1 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%arg1_1,), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %neg_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%neg_1,), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %log1p_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg_3, [3]), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 4), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%div,), kwargs = {})
triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0 = async_compile.triton('triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = triton_helpers.minimum(tmp2, tmp1)
    tmp4 = tl_math.abs(tmp1)
    tmp5 = -tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = libdevice.log1p(tmp6)
    tmp8 = tmp3 - tmp7
    tmp9 = tmp0 * tmp8
    tmp10 = 1.0
    tmp11 = tmp10 - tmp0
    tmp12 = -tmp1
    tmp13 = triton_helpers.minimum(tmp2, tmp12)
    tmp14 = tl_math.abs(tmp12)
    tmp15 = -tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tmp13 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tmp9 + tmp19
    tmp21 = -tmp20
    tmp24 = triton_helpers.minimum(tmp2, tmp23)
    tmp25 = tl_math.abs(tmp23)
    tmp26 = -tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = libdevice.log1p(tmp27)
    tmp29 = tmp24 - tmp28
    tmp30 = tmp22 * tmp29
    tmp31 = tmp10 - tmp22
    tmp32 = -tmp23
    tmp33 = triton_helpers.minimum(tmp2, tmp32)
    tmp34 = tl_math.abs(tmp32)
    tmp35 = -tmp34
    tmp36 = tl_math.exp(tmp35)
    tmp37 = libdevice.log1p(tmp36)
    tmp38 = tmp33 - tmp37
    tmp39 = tmp31 * tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = -tmp40
    tmp42 = tmp21 + tmp41
    tmp45 = triton_helpers.minimum(tmp2, tmp44)
    tmp46 = tl_math.abs(tmp44)
    tmp47 = -tmp46
    tmp48 = tl_math.exp(tmp47)
    tmp49 = libdevice.log1p(tmp48)
    tmp50 = tmp45 - tmp49
    tmp51 = tmp43 * tmp50
    tmp52 = tmp10 - tmp43
    tmp53 = -tmp44
    tmp54 = triton_helpers.minimum(tmp2, tmp53)
    tmp55 = tl_math.abs(tmp53)
    tmp56 = -tmp55
    tmp57 = tl_math.exp(tmp56)
    tmp58 = libdevice.log1p(tmp57)
    tmp59 = tmp54 - tmp58
    tmp60 = tmp52 * tmp59
    tmp61 = tmp51 + tmp60
    tmp62 = -tmp61
    tmp63 = tmp42 + tmp62
    tmp66 = triton_helpers.minimum(tmp2, tmp65)
    tmp67 = tl_math.abs(tmp65)
    tmp68 = -tmp67
    tmp69 = tl_math.exp(tmp68)
    tmp70 = libdevice.log1p(tmp69)
    tmp71 = tmp66 - tmp70
    tmp72 = tmp64 * tmp71
    tmp73 = tmp10 - tmp64
    tmp74 = -tmp65
    tmp75 = triton_helpers.minimum(tmp2, tmp74)
    tmp76 = tl_math.abs(tmp74)
    tmp77 = -tmp76
    tmp78 = tl_math.exp(tmp77)
    tmp79 = libdevice.log1p(tmp78)
    tmp80 = tmp75 - tmp79
    tmp81 = tmp73 * tmp80
    tmp82 = tmp72 + tmp81
    tmp83 = -tmp82
    tmp84 = tmp63 + tmp83
    tmp85 = 0.25
    tmp86 = tmp84 * tmp85
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
    tmp89 = tl.sum(tmp87, 1)[:, None]
    tmp90 = 64.0
    tmp91 = tmp89 / tmp90
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp91, None)
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
        # Topologically Sorted Source Nodes: [output0], Original ATen: [aten.log_sigmoid_forward, aten.mul, aten.rsub, aten.neg, aten.add, aten.sum, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0.run(buf2, arg0_1, arg1_1, 1, 64, grid=grid(1), stream=stream0)
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
