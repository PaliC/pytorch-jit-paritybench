# AOT ID: ['49_forward']
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


# kernel path: inductor_cache/p6/cp6d4hl5a3rwryukamvlavqnnlloqnsao3voj3lbt555f5runy3f.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 1048576*y1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzkju7pxyalzapwjamrwxfxoddye4j5mhcsmqoji2u3nzuyt5t2.py
# Topologically Sorted Source Nodes: [masks, masks_1, x, x_1, x_2], Original ATen: [aten.sigmoid, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   masks => sigmoid
#   masks_1 => relu
#   x => convolution
#   x_1 => add
#   x_2 => relu_1
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_1,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%sigmoid,), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_2, %primals_3, %primals_4, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %relu), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
triton_poi_fused_add_convolution_relu_sigmoid_1 = async_compile.triton('triton_poi_fused_add_convolution_relu_sigmoid_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_sigmoid_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_sigmoid_1(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4096)
    y1 = yindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 4096*x2 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tl.full([1, 1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp2 + tmp6
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mi/cmisjn32ltq3tmnwk6ubqrx2iazsjdwpgji6tk4bf3zbngim6mgq.py
# Topologically Sorted Source Nodes: [x_3, pos], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   pos => relu_2
#   x_3 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_5, %primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
triton_poi_fused_convolution_relu_threshold_backward_2 = async_compile.triton('triton_poi_fused_convolution_relu_threshold_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16384*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp2, ymask)
    tl.store(out_ptr1 + (x2 + 4096*y3), tmp4, ymask)
    tl.store(out_ptr2 + (y0 + 4*x2 + 16384*y1), tmp6, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_2, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_3, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 256, 64, 64), (1048576, 1, 16384, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf0, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 256, 64, 64), (1048576, 1, 16384, 256))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [masks, masks_1, x, x_1, x_2], Original ATen: [aten.sigmoid, aten.relu, aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_sigmoid_1.run(buf2, primals_4, primals_1, 16384, 256, grid=grid(16384, 256), stream=stream0)
        del primals_1
        del primals_4
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 64, 64), (16384, 1, 256, 4))
        buf4 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 64, 64), (16384, 1, 256, 4), torch.bool)
        # Topologically Sorted Source Nodes: [x_3, pos], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_2.run(buf3, primals_6, buf4, buf5, buf6, 16, 4096, grid=grid(16, 4096), stream=stream0)
        del buf3
        del primals_6
    return (buf4, buf5, buf0, primals_3, primals_5, buf2, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
