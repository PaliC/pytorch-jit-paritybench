# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/ct/cctcthhrj6tkkeo7ywbztnujxwoil6s6psuneznnqavug3677fet.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_1,), kwargs = {})
triton_poi_fused_sigmoid_0 = async_compile.triton('triton_poi_fused_sigmoid_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = xindex // 3
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x1), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + (x2), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lg/clg75qeinbopcvejn4jh34juxfqi5ra7z2wmxjduum5nwcuq524x.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%slice_6,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused_relu_threshold_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp3 = tl.load(in_ptr1 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 3, tl.int64)
    tmp1 = tmp0 < tmp0
    tmp2 = tl.load(in_ptr0 + (3 + 3*x2), tmp1 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = 0.0
    tmp8 = tmp6 <= tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vc/cvc7jhq75vwggaw3bco7w5m7esdgtedtgjb4oqiuqfwo2cjmf536.py
# Topologically Sorted Source Nodes: [relu, setitem_1], Original ATen: [aten.relu, aten.copy]
# Source node to ATen node mapping:
#   relu => relu
#   setitem_1 => copy_1
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%slice_6,), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_6, %relu), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%view_7, %copy_1, 3, 3, 9223372036854775807), kwargs = {})
triton_poi_fused_copy_relu_2 = async_compile.triton('triton_poi_fused_copy_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x3 = xindex // 4
    x4 = xindex
    tmp16 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 3, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x0 + 3*x3), tmp6 & xmask, other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (x4), tmp2 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x0 + 3*x3), tmp14 & xmask, other=0.0)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp2, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x4), tmp18, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_h], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((4, 4, 4, 3), (48, 12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_0.run(buf0, buf1, 192, grid=grid(192), stream=stream0)
        buf4 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_1.run(buf1, buf0, buf4, 64, grid=grid(64), stream=stream0)
        buf2 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [relu, setitem_1], Original ATen: [aten.relu, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_relu_2.run(buf2, buf1, 256, grid=grid(256), stream=stream0)
        buf3 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf2, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf3)
        del primals_5
    return (reinterpret_tensor(buf3, (4, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), buf1, reinterpret_tensor(buf2, (64, 4), (4, 1), 0), primals_4, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
