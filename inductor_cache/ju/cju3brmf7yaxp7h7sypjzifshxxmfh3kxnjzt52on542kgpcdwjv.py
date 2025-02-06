# AOT ID: ['36_forward']
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


# kernel path: inductor_cache/zv/czvpf36p22gbgx27aek4vzar477jpcomfc42qya3pyjzvop2rua3.py
# Topologically Sorted Source Nodes: [loc_feature], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   loc_feature => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_8, %primals_9, %primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3936256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3844) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/zu/czuayknfces5maadvxlyl53aqgg46dqvpbqe7sxzzv4lhrxk37rc.py
# Topologically Sorted Source Nodes: [loc_kernel, pk_1], Original ATen: [aten.convolution, aten.view]
# Source node to ATen node mapping:
#   loc_kernel => convolution_1
#   pk_1 => view_3
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_3 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [-1, 256, 62, 62]), kwargs = {})
triton_poi_fused_convolution_view_1 = async_compile.triton('triton_poi_fused_convolution_view_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_view_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15745024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 3844) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/tm/ctm5fh7ijf4dp5bu4sjze3ylrq73qznnz5a55atvan2a2qthkkwi.py
# Topologically Sorted Source Nodes: [loc], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   loc => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_5, %primals_11, %primals_12, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/57/c57l6jfkiq5v367aiopjwupqvk4jn4pqmaksnyirpfdthdlaquyw.py
# Topologically Sorted Source Nodes: [cls_kernel, pk], Original ATen: [aten.convolution, aten.view]
# Source node to ATen node mapping:
#   cls_kernel => convolution
#   pk => view
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [-1, 256, 62, 62]), kwargs = {})
triton_poi_fused_convolution_view_3 = async_compile.triton('triton_poi_fused_convolution_view_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7872512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 3844) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_4, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_5, (1024, ), (1, ))
    assert_size_stride(primals_6, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_7, (256, ), (1, ))
    assert_size_stride(primals_8, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [loc_feature], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_8, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 256, 62, 62), (984064, 3844, 62, 1))
        buf10 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [loc_feature], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf10, primals_10, 3936256, grid=grid(3936256), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [loc_kernel], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_3, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1024, 62, 62), (3936256, 3844, 62, 1))
        buf8 = buf1; del buf1  # reuse
        buf9 = reinterpret_tensor(buf8, (16, 256, 62, 62), (984064, 3844, 62, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [loc_kernel, pk_1], Original ATen: [aten.convolution, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_view_1.run(buf9, primals_5, 15745024, grid=grid(15745024), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [po_2], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(reinterpret_tensor(buf10, (1, 1024, 62, 62), (0, 3844, 62, 1), 0), buf9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf11, (1, 16, 1, 1), (16, 1, 1, 1))
        # Topologically Sorted Source Nodes: [loc], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf11, (4, 4, 1, 1), (4, 1, 1, 1), 0), primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 4, 1, 1), (4, 1, 1, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [loc], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf13, primals_12, 16, grid=grid(16), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [cls_feature], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(primals_8, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 256, 62, 62), (984064, 3844, 62, 1))
        # Topologically Sorted Source Nodes: [cls_kernel], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 512, 62, 62), (1968128, 3844, 62, 1))
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [cls_feature], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf6, primals_7, 3936256, grid=grid(3936256), stream=stream0)
        del primals_7
        buf4 = buf0; del buf0  # reuse
        buf5 = reinterpret_tensor(buf4, (8, 256, 62, 62), (984064, 3844, 62, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [cls_kernel, pk], Original ATen: [aten.convolution, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_view_3.run(buf5, primals_2, 7872512, grid=grid(7872512), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [po], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf6, (1, 1024, 62, 62), (0, 3844, 62, 1), 0), buf5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (1, 8, 1, 1), (8, 1, 1, 1))
    return (reinterpret_tensor(buf7, (4, 2, 1, 1), (2, 1, 1, 1), 0), buf13, primals_1, primals_3, primals_4, primals_6, primals_8, primals_9, primals_11, buf5, reinterpret_tensor(buf6, (1, 1024, 62, 62), (3936256, 3844, 62, 1), 0), buf9, reinterpret_tensor(buf10, (1, 1024, 62, 62), (3936256, 3844, 62, 1), 0), reinterpret_tensor(buf11, (4, 4, 1, 1), (4, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
