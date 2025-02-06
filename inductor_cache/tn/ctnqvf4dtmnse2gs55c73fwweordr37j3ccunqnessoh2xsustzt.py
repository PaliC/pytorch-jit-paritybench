# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/s7/cs73tj7opnvajveqslod4prsioordwfnojxgwwaxn27l66gflwag.py
# Topologically Sorted Source Nodes: [output_x, pow_1, output_y, pow_2, add, add_1, inputs_grad, output_x_1, pow_3, output_y_1, pow_4, add_2, add_3, targets_grad, grad_loss], Original ATen: [aten.constant_pad_nd, aten.pow, aten.add, aten.sqrt, aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   grad_loss => abs_1, mean, sub
#   inputs_grad => sqrt
#   output_x => constant_pad_nd
#   output_x_1 => constant_pad_nd_2
#   output_y => constant_pad_nd_1
#   output_y_1 => constant_pad_nd_3
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   targets_grad => sqrt_1
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convolution, [1, 1, 0, 0], 0.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%constant_pad_nd, 2), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convolution_1, [0, 0, 1, 1], 0.0), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%constant_pad_nd_1, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 0.0001), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1,), kwargs = {})
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convolution_2, [1, 1, 0, 0], 0.0), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%constant_pad_nd_2, 2), kwargs = {})
#   %constant_pad_nd_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convolution_3, [0, 0, 1, 1], 0.0), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%constant_pad_nd_3, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_3, %pow_4), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0001), kwargs = {})
#   %sqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_3,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
triton_red_fused_abs_add_constant_pad_nd_mean_pow_sqrt_sub_0 = async_compile.triton('triton_red_fused_abs_add_constant_pad_nd_mean_pow_sqrt_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_constant_pad_nd_mean_pow_sqrt_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_add_constant_pad_nd_mean_pow_sqrt_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 36
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = (rindex % 64)
        r5 = rindex // 64
        r2 = ((rindex // 64) % 64)
        r3 = rindex // 4096
        r6 = (rindex % 4096)
        r4 = rindex
        tmp0 = (-1) + r1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 62, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + ((-1) + r1 + 62*r5 + 7936*x0), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 * tmp6
        tmp8 = (-1) + r2
        tmp9 = tmp8 >= tmp1
        tmp10 = tmp8 < tmp3
        tmp11 = tmp9 & tmp10
        tmp12 = tl.load(in_ptr1 + ((-64) + r6 + 3968*r3 + 7936*x0), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 * tmp12
        tmp14 = tmp7 + tmp13
        tmp15 = 0.0001
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.sqrt(tmp16)
        tmp18 = tl.load(in_ptr2 + ((-1) + r1 + 62*r5 + 7936*x0), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp18 * tmp18
        tmp20 = tl.load(in_ptr3 + ((-64) + r6 + 3968*r3 + 7936*x0), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp20 * tmp20
        tmp22 = tmp19 + tmp21
        tmp23 = tmp22 + tmp15
        tmp24 = libdevice.sqrt(tmp23)
        tmp25 = tmp17 - tmp24
        tmp26 = tl_math.abs(tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tl.store(out_ptr0 + (r4 + 8192*x0), tmp17, rmask & xmask)
        tl.store(out_ptr1 + (r4 + 8192*x0), tmp24, rmask & xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uc/cucn32d4c7kkwqvvd6i3qol2lr5qh5e33bvtsgy376u46tm6cbn3.py
# Topologically Sorted Source Nodes: [grad_loss, loss], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul]
# Source node to ATen node mapping:
#   grad_loss => abs_1, mean, sub
#   loss => mul
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt, %sqrt_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1), kwargs = {})
triton_per_fused_abs_mean_mul_sub_1 = async_compile.triton('triton_per_fused_abs_mean_mul_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_mean_mul_sub_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 294912.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (18, 6, 1, 3), (18, 3, 3, 1))
    assert_size_stride(arg1_1, (4, 18, 64, 64), (73728, 4096, 64, 1))
    assert_size_stride(arg2_1, (18, 6, 3, 1), (18, 3, 1, 1))
    assert_size_stride(arg3_1, (4, 18, 64, 64), (73728, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [inputs_grad_x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (4, 18, 64, 62), (71424, 3968, 62, 1))
        # Topologically Sorted Source Nodes: [inputs_grad_y], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(arg1_1, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf1, (4, 18, 62, 64), (71424, 3968, 64, 1))
        del arg1_1
        # Topologically Sorted Source Nodes: [targets_grad_x], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(arg3_1, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf3, (4, 18, 64, 62), (71424, 3968, 62, 1))
        del arg0_1
        # Topologically Sorted Source Nodes: [targets_grad_y], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(arg3_1, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf4, (4, 18, 62, 64), (71424, 3968, 64, 1))
        del arg2_1
        del arg3_1
        buf2 = empty_strided_cuda((4, 18, 64, 64), (73728, 4096, 64, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 18, 64, 64), (73728, 4096, 64, 1), torch.float32)
        buf6 = empty_strided_cuda((36, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [output_x, pow_1, output_y, pow_2, add, add_1, inputs_grad, output_x_1, pow_3, output_y_1, pow_4, add_2, add_3, targets_grad, grad_loss], Original ATen: [aten.constant_pad_nd, aten.pow, aten.add, aten.sqrt, aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_constant_pad_nd_mean_pow_sqrt_sub_0.run(buf0, buf1, buf3, buf4, buf2, buf5, buf6, 36, 8192, grid=grid(36), stream=stream0)
        del buf0
        del buf1
        del buf3
        del buf4
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [grad_loss, loss], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_mean_mul_sub_1.run(buf8, buf6, 1, 36, grid=grid(1), stream=stream0)
        del buf6
    return (buf8, buf2, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((18, 6, 1, 3), (18, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 18, 64, 64), (73728, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((18, 6, 3, 1), (18, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 18, 64, 64), (73728, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
