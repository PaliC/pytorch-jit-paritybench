# AOT ID: ['149_forward']
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


# kernel path: inductor_cache/lk/clkcdlsrmrmiajdstjxojv6lddi23wze76hxth5ginls5okzf74f.py
# Topologically Sorted Source Nodes: [t, t_1, t_2, t_3], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   t => relu
#   t_1 => relu_1
#   t_2 => relu_2
#   t_3 => relu_3
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze,), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_3,), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_6,), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_9,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'out_ptr2': '*i1', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp11 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp15 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp8 = tmp7 + tmp1
    tmp9 = triton_helpers.maximum(tmp3, tmp8)
    tmp10 = tmp9 <= tmp5
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp3, tmp12)
    tmp14 = tmp13 <= tmp5
    tmp16 = tmp15 + tmp1
    tmp17 = triton_helpers.maximum(tmp3, tmp16)
    tmp18 = tmp17 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(in_out_ptr1 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp10, xmask)
    tl.store(in_out_ptr2 + (x2), tmp13, xmask)
    tl.store(out_ptr2 + (x2), tmp14, xmask)
    tl.store(in_out_ptr3 + (x2), tmp17, xmask)
    tl.store(out_ptr3 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6e/c6eir3oc3m3qyztlvmdq7nefhmhxo5mdtryztzccuwlbgv6bz5be.py
# Topologically Sorted Source Nodes: [conv2d_1, conv2d_4, conv2d_7, conv2d_10], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   conv2d_10 => convolution_10
#   conv2d_4 => convolution_4
#   conv2d_7 => convolution_7
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_4, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_7, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_10, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp7 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp4, xmask)
    tl.store(in_out_ptr2 + (x2), tmp6, xmask)
    tl.store(in_out_ptr3 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2h/c2h6cfcrbsvh2m2n2pjz46qvvca6jd4x5amncq5s4e4qw4zmz5mq.py
# Topologically Sorted Source Nodes: [conv2d_2, conv2d_5, conv2d_8, conv2d_11], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_11 => convolution_11
#   conv2d_2 => convolution_2
#   conv2d_5 => convolution_5
#   conv2d_8 => convolution_8
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_1, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_4, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_7, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_10, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp7 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp4, xmask)
    tl.store(in_out_ptr2 + (x2), tmp6, xmask)
    tl.store(in_out_ptr3 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_7, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 0), primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 128), primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 192), primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 64), primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (1, 4, 4, 4), (64, 16, 4, 1))
        buf1 = reinterpret_tensor(buf0, (4, 4, 4), (16, 4, 1), 0); del buf0  # reuse
        buf27 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        buf7 = reinterpret_tensor(buf6, (4, 4, 4), (16, 4, 1), 0); del buf6  # reuse
        buf26 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        buf13 = reinterpret_tensor(buf12, (4, 4, 4), (16, 4, 1), 0); del buf12  # reuse
        buf25 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        buf19 = reinterpret_tensor(buf18, (4, 4, 4), (16, 4, 1), 0); del buf18  # reuse
        buf24 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [t, t_1, t_2, t_3], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf1, buf7, buf13, buf19, primals_3, buf27, buf26, buf25, buf24, 64, grid=grid(64), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(buf1, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (1, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf7, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1, 4, 4, 4), (64, 16, 4, 1))
        buf3 = buf2; del buf2  # reuse
        buf9 = buf8; del buf8  # reuse
        buf15 = buf14; del buf14  # reuse
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, conv2d_4, conv2d_7, conv2d_10], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf3, buf9, buf15, buf21, primals_5, 64, grid=grid(64), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf1, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (1, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf7, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(reinterpret_tensor(buf13, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (1, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(reinterpret_tensor(buf19, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (1, 16, 4, 4), (256, 16, 4, 1))
        buf5 = buf4; del buf4  # reuse
        buf11 = buf10; del buf10  # reuse
        buf17 = buf16; del buf16  # reuse
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2, conv2d_5, conv2d_8, conv2d_11], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf5, buf11, buf17, buf23, primals_7, 256, grid=grid(256), stream=stream0)
        del primals_7
    return (reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf9, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf15, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf21, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf5, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf11, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf17, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf23, (16, 4, 4), (16, 4, 1), 0), primals_2, primals_4, primals_6, reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(buf1, (1, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 64), reinterpret_tensor(buf7, (1, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 128), reinterpret_tensor(buf13, (1, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 192), reinterpret_tensor(buf19, (1, 4, 4, 4), (64, 16, 4, 1), 0), buf24, buf25, buf26, buf27, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
