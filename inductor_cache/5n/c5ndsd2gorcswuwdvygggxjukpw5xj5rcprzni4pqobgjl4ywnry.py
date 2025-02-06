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


# kernel path: inductor_cache/5b/c5bwyxtepns2j5umskam7iyotvxprswvcrtzlog7vhtvno2erral.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div, mul, pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2, 3], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %pow_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
triton_per_fused__weight_norm_interface_0 = async_compile.triton('triton_per_fused__weight_norm_interface_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 36*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 36*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4f/c4ff2w65jkpil3bnjdmwwvq37xpr4jxkfvfxc7wovjjl4z6ywsag.py
# Topologically Sorted Source Nodes: [conv2d, sub, relu, x], Original ATen: [aten.convolution, aten.sub, aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   conv2d => convolution
#   relu => relu
#   sub => sub
#   x => add
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %mul, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %primals_5), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%sub,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %primals_5), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_add_convolution_relu_sub_threshold_backward_1 = async_compile.triton('triton_poi_fused_add_convolution_relu_sub_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_sub_threshold_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_sub_threshold_backward_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tmp7 + tmp4
    tmp9 = 0.0
    tmp10 = tmp7 <= tmp9
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tm/ctmodlrd6rnfbfpt6522k3gsgd7sa6ubtechz3csg4e7ygu6e4mr.py
# Topologically Sorted Source Nodes: [out, out_1, sub_1, relu_1, x_1], Original ATen: [aten.convolution, aten.add, aten.sub, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out => convolution_1
#   out_1 => add_1
#   relu_1 => relu_1
#   sub_1 => sub_1
#   x_1 => add_2
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add, %mul_1, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %primals_4), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %primals_9), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%sub_1,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_1, %primals_9), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
triton_poi_fused_add_convolution_relu_sub_threshold_backward_2 = async_compile.triton('triton_poi_fused_add_convolution_relu_sub_threshold_backward_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_sub_threshold_backward_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_sub_threshold_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp4 - tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tmp9 + tmp6
    tmp11 = 0.0
    tmp12 = tmp9 <= tmp11
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_5, (1, ), (1, ))
    assert_size_stride(primals_6, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf1, primals_2, primals_1, buf2, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_4, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 16, 4, 1))
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d, sub, relu, x], Original ATen: [aten.convolution, aten.sub, aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_sub_threshold_backward_1.run(buf3, primals_3, primals_5, buf4, buf11, 256, grid=grid(256), stream=stream0)
        del primals_3
        del primals_5
        buf5 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf6 = reinterpret_tensor(buf5, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf5  # reuse
        buf7 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf6, primals_7, primals_6, buf7, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf4, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 4, 4), (64, 16, 4, 1))
        buf9 = buf3; del buf3  # reuse
        buf10 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out, out_1, sub_1, relu_1, x_1], Original ATen: [aten.convolution, aten.add, aten.sub, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_sub_threshold_backward_2.run(buf8, primals_8, primals_4, primals_9, buf9, buf10, 256, grid=grid(256), stream=stream0)
        del buf8
        del primals_8
        del primals_9
    return (buf9, buf2, buf7, primals_1, primals_2, primals_4, primals_6, primals_7, buf1, buf2, buf4, buf6, buf7, buf10, buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
