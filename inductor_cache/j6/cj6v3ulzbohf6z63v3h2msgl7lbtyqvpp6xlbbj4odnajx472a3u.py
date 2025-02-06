# AOT ID: ['7_inference']
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


# kernel path: inductor_cache/33/c33fd7riw4kdinrdlytocew3llihprvangkjczougkjq3h2jtbzc.py
# Topologically Sorted Source Nodes: [target_mean, target], Original ATen: [aten.mean, aten.sub]
# Source node to ATen node mapping:
#   target => sub_1
#   target_mean => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg1_1, [-1], True), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %mean_1), kwargs = {})
triton_poi_fused_mean_sub_0 = async_compile.triton('triton_poi_fused_mean_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp0 - tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gh/cghmju5pkumtnce3qxkmbukc2n6cdp7wwuafbjeoxy2pqk6em2uw.py
# Topologically Sorted Source Nodes: [pow_1, sum_1, input_mean, input_1, res, pow_2, sum_2, add, truediv, add_1, log10, losses, losses_1, neg], Original ATen: [aten.pow, aten.sum, aten.mean, aten.sub, aten.add, aten.div, aten.log10, aten.mul, aten.neg]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   input_1 => sub
#   input_mean => mean
#   log10 => log10
#   losses => mul
#   losses_1 => mean_2
#   neg => neg
#   pow_1 => pow_1
#   pow_2 => pow_2
#   res => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   truediv => div
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg0_1, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %mean), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %add), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, 1e-08), kwargs = {})
#   %log10 : [num_users=1] = call_function[target=torch.ops.aten.log10.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%log10, 10), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mean_2,), kwargs = {})
triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1 = async_compile.triton('triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 4.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp11 - tmp19
    tmp21 = tmp20 - tmp0
    tmp22 = tmp21 * tmp21
    tmp23 = tmp12 - tmp19
    tmp24 = tmp23 - tmp2
    tmp25 = tmp24 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tmp14 - tmp19
    tmp28 = tmp27 - tmp5
    tmp29 = tmp28 * tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tmp16 - tmp19
    tmp32 = tmp31 - tmp8
    tmp33 = tmp32 * tmp32
    tmp34 = tmp30 + tmp33
    tmp35 = 1e-08
    tmp36 = tmp34 + tmp35
    tmp37 = tmp10 / tmp36
    tmp38 = tmp37 + tmp35
    tmp39 = libdevice.log10(tmp38)
    tmp40 = 10.0
    tmp41 = tmp39 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
    tmp44 = tl.sum(tmp42, 1)[:, None]
    tmp45 = 64.0
    tmp46 = tmp44 / tmp45
    tmp47 = -tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp47, None)
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
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target_mean, target], Original ATen: [aten.mean, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_sub_0.run(arg1_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [pow_1, sum_1, input_mean, input_1, res, pow_2, sum_2, add, truediv, add_1, log10, losses, losses_1, neg], Original ATen: [aten.pow, aten.sum, aten.mean, aten.sub, aten.add, aten.div, aten.log10, aten.mul, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1.run(buf3, buf0, arg0_1, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del buf0
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
