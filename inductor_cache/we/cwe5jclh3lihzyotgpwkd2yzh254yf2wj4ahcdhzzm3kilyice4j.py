# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/mm/cmmwipghgt256u2cymbfj2rvoxcnqyvnbzqid5wflk3gsx2rg5hz.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_2 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gl/cglrkfeiqad3ujtq4uoa3wylbw7fhy6kf7t2nzx7xegsxjkkkam2.py
# Topologically Sorted Source Nodes: [input_7, sub, pow_1, neg, exp, positive, sub_1, pow_2, neg_1, negative, sum_1, sum_2, sub_2, upper_bound, truediv_2], Original ATen: [aten.tanh, aten.sub, aten.pow, aten.neg, aten.exp, aten.div, aten.sum, aten.mean]
# Source node to ATen node mapping:
#   exp => exp
#   input_7 => tanh
#   neg => neg
#   neg_1 => neg_1
#   negative => div_1
#   positive => div
#   pow_1 => pow_1
#   pow_2 => pow_2
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   truediv_2 => div_2
#   upper_bound => mean
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%view_7,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %primals_10), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%pow_1,), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%tanh,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg, %exp), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %index), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%pow_2,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg_1, %exp), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div, [-1]), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_1, [-1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_2,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mean, 2.0), kwargs = {})
triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1 = async_compile.triton('triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = -tmp3
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp4 / tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = -tmp12
    tmp15 = libdevice.tanh(tmp14)
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp13 / tmp16
    tmp18 = tmp8 + tmp17
    tmp21 = tmp19 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = -tmp22
    tmp25 = libdevice.tanh(tmp24)
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp23 / tmp26
    tmp28 = tmp18 + tmp27
    tmp31 = tmp29 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = -tmp32
    tmp35 = libdevice.tanh(tmp34)
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp33 / tmp36
    tmp38 = tmp28 + tmp37
    tmp40 = tmp0 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = -tmp41
    tmp43 = tmp42 / tmp7
    tmp45 = tmp9 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = -tmp46
    tmp48 = tmp47 / tmp16
    tmp49 = tmp43 + tmp48
    tmp51 = tmp19 - tmp50
    tmp52 = tmp51 * tmp51
    tmp53 = -tmp52
    tmp54 = tmp53 / tmp26
    tmp55 = tmp49 + tmp54
    tmp57 = tmp29 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = -tmp58
    tmp60 = tmp59 / tmp36
    tmp61 = tmp55 + tmp60
    tmp62 = tmp38 - tmp61
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
    tmp65 = tl.sum(tmp63, 1)[:, None]
    tmp66 = 64.0
    tmp67 = tmp65 / tmp66
    tmp68 = 0.5
    tmp69 = tmp67 * tmp68
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp69, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (2, 4), (4, 1))
    assert_size_stride(primals_2, (2, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 2), (2, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (2, 4), (4, 1))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (4, 2), (2, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 2), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 2), (32, 8, 2, 1), 0); del buf0  # reuse
        buf14 = empty_strided_cuda((4, 4, 4, 2), (32, 8, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf1, primals_2, buf14, 128, grid=grid(128), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf1, (64, 2), (2, 1), 0), reinterpret_tensor(primals_4, (2, 4), (1, 2), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        buf3 = empty_strided_cuda((64, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 2), (1, 4), 0), out=buf3)
        del primals_6
        buf4 = reinterpret_tensor(buf3, (4, 4, 4, 2), (32, 8, 2, 1), 0); del buf3  # reuse
        buf13 = empty_strided_cuda((4, 4, 4, 2), (32, 8, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf4, primals_7, buf13, 128, grid=grid(128), stream=stream0)
        del primals_7
        buf5 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf4, (64, 2), (2, 1), 0), reinterpret_tensor(primals_8, (2, 4), (1, 2), 0), alpha=1, beta=1, out=buf5)
        del primals_9
    # Topologically Sorted Source Nodes: [randperm], Original ATen: [aten.randperm]
    buf6 = torch.ops.aten.randperm.default(4, device=device(type='cpu'), pin_memory=False)
    buf7 = buf6
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.index]
        buf8 = torch.ops.aten.index.Tensor(primals_10, [buf7])
        buf9 = buf8
        del buf8
        buf12 = empty_strided_cuda((), (), torch.float32)
        buf15 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_7, sub, pow_1, neg, exp, positive, sub_1, pow_2, neg_1, negative, sum_1, sum_2, sub_2, upper_bound, truediv_2], Original ATen: [aten.tanh, aten.sub, aten.pow, aten.neg, aten.exp, aten.div, aten.sum, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1.run(buf15, buf2, primals_10, buf5, buf9, 1, 64, grid=grid(1), stream=stream0)
        del buf9
    return (buf15, primals_10, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(buf1, (64, 2), (2, 1), 0), buf2, reinterpret_tensor(buf4, (64, 2), (2, 1), 0), buf5, buf7, primals_8, buf13, primals_4, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
