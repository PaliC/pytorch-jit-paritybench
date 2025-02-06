# AOT ID: ['10_forward']
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


# kernel path: inductor_cache/n3/cn3j4bl2f467yhqwkafelignlb5372ji5dgzradeobzdc3vo2vkj.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_poi_fused_convolution_relu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6v/c6vkvyosjl2z5hdumv5mxcrbor7wns7kbjdhnkxozyxisbrhnws2.py
# Topologically Sorted Source Nodes: [input_5, y], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_5 => convolution_2
#   y => add
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_6, %primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %relu), kwargs = {})
triton_poi_fused_add_convolution_1 = async_compile.triton('triton_poi_fused_add_convolution_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvxx7u5lrcx255edizcrffd3we2ekcltx4ofuyjqxgsc3fqdzpe.py
# Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_57 => convolution_36
# Graph fragment:
#   %convolution_36 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_15, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf1, primals_2, 256, grid=grid(256), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 16, 4, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf3, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 4, 4), (64, 16, 4, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_5, y], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf5, primals_7, buf1, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 4, 4), (64, 16, 4, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf7, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 4, 4), (64, 16, 4, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_8, y_1], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf9, primals_7, buf5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf11, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 4, 4, 4), (64, 16, 4, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_11, y_2], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf13, primals_7, buf9, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 4, 4), (64, 16, 4, 1))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf15, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 4, 4, 4), (64, 16, 4, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_14, y_3], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf17, primals_7, buf13, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 4, 4, 4), (64, 16, 4, 1))
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf19, primals_9, 256, grid=grid(256), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf21, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 4, 4), (64, 16, 4, 1))
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_19, y_4], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf23, primals_7, buf19, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 4, 4), (64, 16, 4, 1))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf25, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 4, 4, 4), (64, 16, 4, 1))
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_22, y_5], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf27, primals_7, buf23, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 4, 4), (64, 16, 4, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf29, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 4, 4, 4), (64, 16, 4, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_25, y_6], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf31, primals_7, buf27, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4), (64, 16, 4, 1))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf33, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 4, 4, 4), (64, 16, 4, 1))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_28, y_7], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf35, primals_7, buf31, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 4, 4), (64, 16, 4, 1))
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf37, primals_11, 256, grid=grid(256), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 4, 4, 4), (64, 16, 4, 1))
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf39, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_33, y_8], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf41, primals_7, buf37, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 4, 4, 4), (64, 16, 4, 1))
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf43, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 4, 4, 4), (64, 16, 4, 1))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_36, y_9], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf45, primals_7, buf41, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 4, 4, 4), (64, 16, 4, 1))
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf47, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 4, 4, 4), (64, 16, 4, 1))
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_39, y_10], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf49, primals_7, buf45, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 4, 4), (64, 16, 4, 1))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf51, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 4, 4, 4), (64, 16, 4, 1))
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_42, y_11], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf53, primals_7, buf49, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 4, 4, 4), (64, 16, 4, 1))
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf55, primals_13, 256, grid=grid(256), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 4, 4, 4), (64, 16, 4, 1))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf57, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 4, 4, 4), (64, 16, 4, 1))
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_47, y_12], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf59, primals_7, buf55, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 4, 4, 4), (64, 16, 4, 1))
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf61, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 4, 4, 4), (64, 16, 4, 1))
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_50, y_13], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf63, primals_7, buf59, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 4, 4, 4), (64, 16, 4, 1))
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf65, primals_5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 4, 4, 4), (64, 16, 4, 1))
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_53, y_14], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf67, primals_7, buf63, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 4, 4, 4), (64, 16, 4, 1))
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf69, primals_5, 256, grid=grid(256), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 4, 4, 4), (64, 16, 4, 1))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_56, y_15], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_1.run(buf71, primals_7, buf67, 256, grid=grid(256), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 4, 4, 4), (64, 16, 4, 1))
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf73, primals_15, 256, grid=grid(256), stream=stream0)
        del primals_15
    return (buf73, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
