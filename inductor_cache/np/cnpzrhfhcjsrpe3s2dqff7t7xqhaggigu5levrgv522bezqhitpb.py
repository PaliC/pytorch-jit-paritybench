# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/kj/ckjbzkgzgq7lxqtmziya3g7lirrcidfolhelqojk3p2br4gc7f6j.py
# Topologically Sorted Source Nodes: [instance_norm, output], Original ATen: [aten._native_batch_norm_legit, aten.leaky_relu]
# Source node to ATen node mapping:
#   instance_norm => add, rsqrt, var_mean
#   output => gt, mul_1, where
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 0.001), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_1, %mul_1), kwargs = {})
triton_red_fused__native_batch_norm_legit_leaky_relu_0 = async_compile.triton('triton_red_fused__native_batch_norm_legit_leaky_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_leaky_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_leaky_relu_0(in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 4096.0
        tmp8 = tmp3 / tmp7
        tmp9 = 0.001
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp13 = 0.0
        tmp14 = tmp12 > tmp13
        tmp15 = 0.2
        tmp16 = tmp12 * tmp15
        tmp17 = tl.where(tmp14, tmp12, tmp16)
        tl.store(out_ptr2 + (r1 + 4096*x0), tmp17, rmask & xmask)
    tmp18 = 4096.0
    tmp19 = tmp3 / tmp18
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3s/c3svbasefkv2mrm32hy7tznihjlrbiyzexph4zo2hgf5kwc6c4yj.py
# Topologically Sorted Source Nodes: [instance_norm_1, output_2], Original ATen: [aten._native_batch_norm_legit, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   instance_norm_1 => add_2, rsqrt_1, var_mean_1
#   output_2 => gt_1, mul_3, where_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 0.001), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_6, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, 0.2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %view_6, %mul_3), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_13, 0), kwargs = {})
triton_red_fused__native_batch_norm_legit_leaky_relu_leaky_relu_backward_1 = async_compile.triton('triton_red_fused__native_batch_norm_legit_leaky_relu_leaky_relu_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_leaky_relu_leaky_relu_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_leaky_relu_leaky_relu_backward_1(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 4096.0
        tmp12 = tmp5 / tmp11
        tmp13 = 0.001
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = 0.0
        tmp18 = tmp16 > tmp17
        tmp19 = 0.2
        tmp20 = tmp16 * tmp19
        tmp21 = tl.where(tmp18, tmp16, tmp20)
        tmp22 = tmp21 > tmp17
        tl.store(out_ptr2 + (r1 + 4096*x0), tmp21, rmask & xmask)
        tl.store(out_ptr3 + (r1 + 4096*x0), tmp22, rmask & xmask)
    tmp23 = 4096.0
    tmp24 = tmp5 / tmp23
    tmp25 = 0.001
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_2, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf5 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm, output], Original ATen: [aten._native_batch_norm_legit, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_leaky_relu_0.run(buf0, buf1, buf5, buf4, 256, 4096, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf7 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf11 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.bool)
        buf10 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_1, output_2], Original ATen: [aten._native_batch_norm_legit, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_leaky_relu_leaky_relu_backward_1.run(buf6, primals_1, buf7, buf11, buf12, buf10, 256, 4096, grid=grid(256), stream=stream0)
    return (buf11, primals_1, primals_2, primals_3, buf0, reinterpret_tensor(buf4, (256, ), (1, ), 0), buf5, buf6, reinterpret_tensor(buf10, (256, ), (1, ), 0), buf12, reinterpret_tensor(buf7, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf1, (1, 256, 1, 1), (256, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
