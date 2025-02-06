# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/qt/cqtiruaf3kekvcs2xdegmpin3iwpxtyfrscwelp4e6aalahwy773.py
# Topologically Sorted Source Nodes: [conv2d, conv1], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv1 => gt, mul, where
#   conv2d => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
triton_poi_fused_convolution_leaky_relu_0 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5mvlv2chvyffc3o6rsp4wayckgtjwg33vlh2wxgyietrbkfbkpi.py
# Topologically Sorted Source Nodes: [conv2, out], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   conv2 => convolution_1
#   out => add
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %convolution_1), kwargs = {})
triton_poi_fused_add_convolution_1 = async_compile.triton('triton_poi_fused_add_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 32, 64, 64), (131072, 4096, 64, 1))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [conv2d, conv1], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_0.run(buf1, primals_2, 524288, grid=grid(524288), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [conv2], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [conv2, out], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf3, primals_3, primals_5, 524288, grid=grid(524288), stream=stream0)
        del primals_5
    return (buf3, primals_1, primals_3, primals_4, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 32, 64, 64), (131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
