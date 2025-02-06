# AOT ID: ['33_inference']
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


# kernel path: inductor_cache/k7/ck7puy6axkf25sf5rgwfawhaahf4bbuhv6cuakgjvb7qul3k5cl3.py
# Topologically Sorted Source Nodes: [mu1, mu2, mul_3, mul_1, mul_2], Original ATen: [aten.convolution, aten.mul]
# Source node to ATen node mapping:
#   mu1 => convolution
#   mu2 => convolution_1
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, None, [1, 1], [5, 5], [1, 1], False, [0, 0], 4), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg1_1, None, [1, 1], [5, 5], [1, 1], False, [0, 0], 4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg0_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg2_1), kwargs = {})
triton_poi_fused_convolution_mul_0 = async_compile.triton('triton_poi_fused_convolution_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask & ymask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp1 * tmp1
    tl.store(out_ptr0 + (y0 + 4*x2 + 64*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 4*x2 + 64*y1), tmp1, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 4*x2 + 64*y1), tmp2, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 4*x2 + 64*y1), tmp3, xmask & ymask)
    tl.store(out_ptr4 + (y0 + 4*x2 + 64*y1), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cq/ccqpnuz6vv3x7fzbfl7vt4doxct7rds3pfxmgx5ywowplv7savu6.py
# Topologically Sorted Source Nodes: [mu1_mu2, mul_4, add, sigma12, mul_5, add_1, mul_6, mu1_sq, mu2_sq, add_2, add_3, sigma1_sq, sigma2_sq, add_4, add_5, mul_7, ssim_map, min_1, sub_3, max_1, min_2, sub_4, ssim_map_1, add_6, log, ssim_map_2, mean], Original ATen: [aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.min, aten.max, aten.log, aten.neg, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   log => log
#   max_1 => max_1
#   mean => mean
#   min_1 => min_1
#   min_2 => min_2
#   mu1_mu2 => mul
#   mu1_sq => pow_1
#   mu2_sq => pow_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   sigma12 => sub_2
#   sigma1_sq => sub
#   sigma2_sq => sub_1
#   ssim_map => div
#   ssim_map_1 => div_1
#   ssim_map_2 => neg
#   sub_3 => sub_3
#   sub_4 => sub_4
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %convolution_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.0001), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %mul), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0009), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add_1), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_1, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0001), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %pow_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %pow_2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, 0.0009), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %add_5), kwargs = {})
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %min_1 : [num_users=1] = call_function[target=torch.ops.aten.min.default](args = (%div,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %min_1), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%div,), kwargs = {})
#   %min_2 : [num_users=1] = call_function[target=torch.ops.aten.min.default](args = (%div,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%max_1, %min_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, %sub_4), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, 1e-08), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_6,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%neg,), kwargs = {})
triton_per_fused_add_div_log_max_mean_min_mul_neg_pow_sub_1 = async_compile.triton('triton_per_fused_add_div_log_max_mean_min_mul_neg_pow_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_max_mean_min_mul_neg_pow_sub_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_max_mean_min_mul_neg_pow_sub_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr0 + (r0), None)
    tmp7 = tl.load(in_ptr1 + (r0), None)
    tmp17 = tl.load(in_ptr2 + (r0), None)
    tmp19 = tl.load(in_ptr3 + (r0), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0001
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 - tmp2
    tmp9 = tmp8 * tmp3
    tmp10 = 0.0009
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp0 * tmp0
    tmp14 = tmp1 * tmp1
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 + tmp5
    tmp18 = tmp17 - tmp13
    tmp20 = tmp19 - tmp14
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21 + tmp10
    tmp23 = tmp16 * tmp22
    tmp24 = tmp12 / tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp25, 0))
    tmp29 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp25, 0))
    tmp30 = tmp24 - tmp27
    tmp31 = tmp29 - tmp27
    tmp32 = tmp30 / tmp31
    tmp33 = 1e-08
    tmp34 = tmp32 + tmp33
    tmp35 = tl_math.log(tmp34)
    tmp36 = -tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp40 = 256.0
    tmp41 = tmp39 / tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([1], 0, tl.int32)), tmp41, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 1, 11, 11), (121, 121, 11, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mu1, mu2, mul_3, mul_1, mul_2], Original ATen: [aten.convolution, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_0.run(arg0_1, arg2_1, buf0, buf2, buf4, buf6, buf8, 16, 16, grid=grid(16, 16), stream=stream0)
        del arg0_1
        del arg2_1
        # Topologically Sorted Source Nodes: [mu1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf0
        # Topologically Sorted Source Nodes: [mu2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf2
        # Topologically Sorted Source Nodes: [mul_3, conv2d_4], Original ATen: [aten.mul, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf5, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf4
        # Topologically Sorted Source Nodes: [mul_1, conv2d_2], Original ATen: [aten.mul, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf6
        # Topologically Sorted Source Nodes: [mul_2, conv2d_3], Original ATen: [aten.mul, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf9, (4, 4, 4, 4), (64, 1, 16, 4))
        del arg1_1
        del buf8
        buf10 = buf1; del buf1  # reuse
        buf11 = empty_strided_cuda((), (), torch.float32)
        buf14 = buf11; del buf11  # reuse
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [mu1_mu2, mul_4, add, sigma12, mul_5, add_1, mul_6, mu1_sq, mu2_sq, add_2, add_3, sigma1_sq, sigma2_sq, add_4, add_5, mul_7, ssim_map, min_1, sub_3, max_1, min_2, sub_4, ssim_map_1, add_6, log, ssim_map_2, mean], Original ATen: [aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.min, aten.max, aten.log, aten.neg, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_log_max_mean_min_mul_neg_pow_sub_1.run(buf10, buf15, buf3, buf5, buf7, buf9, 1, 256, grid=grid(1), stream=stream0)
        del buf10
        del buf3
        del buf5
        del buf7
        del buf9
    return (buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 1, 11, 11), (121, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
