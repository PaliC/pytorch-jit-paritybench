# AOT ID: ['18_forward']
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


# kernel path: inductor_cache/lg/clgzst2d5ddg6nvnwx6p72p7cgv2e6ep4sjfvvkcg2xqpopobngu.py
# Topologically Sorted Source Nodes: [_x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   _x => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %mul], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 19)
    x0 = (xindex % 16)
    x2 = xindex // 304
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 48*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 19, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = (-3) + x1
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 15, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tmp13 & tmp6
    tmp15 = 0.0
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tmp9 >= tmp12
    tmp19 = tl.full([1], 16, tl.int64)
    tmp20 = tmp9 < tmp19
    tmp21 = tmp18 & tmp6
    tmp22 = 0.0
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp13, tmp17, tmp24)
    tmp26 = 1.0
    tmp27 = tmp26 * tmp25
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp6, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp5, tmp29)
    tl.store(out_ptr0 + (x3), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3n/c3n3kp4sxchglhdlp23wxkzobyd4jp4jr32wewnprirqqd3xyaax.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7h2lfnbh4xkigtdvodn74teqg7oiwbp767dusslsrh2gizwwqsb.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.pixel_shuffle]
# Source node to ATen node mapping:
#   input_24 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_pixel_shuffle_2 = async_compile.triton('triton_poi_fused_pixel_shuffle_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pixel_shuffle_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pixel_shuffle_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y0 = (yindex % 4)
    y1 = ((yindex // 4) % 4)
    y2 = ((yindex // 16) % 4)
    y6 = yindex // 64
    y3 = ((yindex // 64) % 3)
    y7 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*y2 + 16*x5 + 64*y1 + 256*y6), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + 4*y1 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x5 + 4*y7), tmp2, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (128, 19, 3, 3), (171, 9, 3, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (48, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_25, (48, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 19, 4, 4), (304, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_1, buf0, 1216, grid=grid(1216), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf2, primals_3, 8192, grid=grid(8192), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf4, primals_5, 8192, grid=grid(8192), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf6, primals_7, 8192, grid=grid(8192), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf8, primals_9, 8192, grid=grid(8192), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf10, primals_11, 8192, grid=grid(8192), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf12, primals_13, 8192, grid=grid(8192), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf14, primals_15, 8192, grid=grid(8192), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf16, primals_17, 8192, grid=grid(8192), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf18, primals_19, 8192, grid=grid(8192), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf20, primals_21, 8192, grid=grid(8192), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 128, 4, 4), (2048, 16, 4, 1))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf22, primals_23, 8192, grid=grid(8192), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 48, 4, 4), (768, 16, 4, 1))
        buf24 = empty_strided_cuda((4, 3, 4, 4, 4, 4), (768, 256, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.pixel_shuffle]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pixel_shuffle_2.run(buf23, primals_25, buf24, 768, 4, grid=grid(768, 4), stream=stream0)
        del buf23
        del primals_25
    return (reinterpret_tensor(buf24, (4, 3, 16, 16), (768, 256, 16, 1), 0), primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, buf0, buf2, buf4, buf6, buf8, buf10, buf12, buf14, buf16, buf18, buf20, buf22, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 19, 3, 3), (171, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
