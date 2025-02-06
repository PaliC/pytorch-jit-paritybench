# AOT ID: ['5_inference']
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


# kernel path: inductor_cache/ow/cowslhbp6dpvpu6fmftoqq6nd7nxrdrmt57rh2kw3sghz6wtt7if.py
# Topologically Sorted Source Nodes: [input_1, input_24], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_24 => convolution_10
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/oz/coz5y63s5kjxr2ulna6hj2o2njoitgmgpwhvbjp7b3epk34qlswj.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwroflrevd2gabyxoh7k2aaht2q5xvgl7hh5aydv5l36wbc62n6q.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_3 => convolution_1
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5e/c5enlrkrdexywlgmjthglg3jdjpm4jhtn2scpjxpg2v5cjn6q27y.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pa/cpackjszpusewmfevxrcsojawa6ahhtpjphlx5misezl67p4qyt4.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjivo74gearyy7a5liygr2owzkhi6volwpqngjlocqvm5n3a5gb.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hj/chjj5eq7rowjnwjw35vi42cps4ftbcdq2ldzzeztakzk2ogo5kus.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_6 = async_compile.triton('triton_poi_fused_convolution_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/kh/ckhgjcfn4wzhyaxhvv7xfa73ypm7zurzpbqdb5niuwb4wxy6ubx6.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 32)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 8192*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 8192*x2), None)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 128*x1 + 8192*x2), None)
    tmp5 = tl.load(in_ptr0 + (4160 + x0 + 128*x1 + 8192*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjraf35tydc7ub3q4ugoxf72ptyinqejnrnrujgffyroljqz7ivw.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/wi/cwigk24g7k5aehv7tusz7yxvv4qhdmtcmhcvit2lbav6jw6mzy27.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 8192*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 8192*x2), None)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 256*x1 + 8192*x2), None)
    tmp5 = tl.load(in_ptr0 + (4224 + x0 + 256*x1 + 8192*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/sw/cswwq6dq3cbj6uvs3uudkkkmleqlcpi7cycysztbfn5brdwy4xat.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxqcmqifwlmotavnn2ut67d44wolxhkfqv6nbw5utomdn4gjqgnr.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_11(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zv/czve57ckjgrtzrcsupp2xzsb4zgxipwlnhrt5uowgn2wgq2txac2.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 8)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 8192*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 8192*x2), None)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 512*x1 + 8192*x2), None)
    tmp5 = tl.load(in_ptr0 + (4352 + x0 + 512*x1 + 8192*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/7p/c7pk5ruk4loesabbv53yuzvxh32nazafacy3d2oyxheuxgrnvrzp.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_2 => relu
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_13(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ba/cbagtxg6uhi3e6d52ldcilocagc6hkydhhfyj7yuequgy2sjkjmi.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_14 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/az/cazpt34sxkycfheg57vki2sb5y2tckygfuxmxlvhbnu4lvu5attt.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_41, input_42, input_43], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_15(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ne/cneu5soynpjk6yfdqityslk3vvqs7wrydajkf7757ee3o3fgh4wj.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   mean => mean
#   sub => sub
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_9, %relu_19), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16 = async_compile.triton('triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (512*((r2 % 64)) + 32768*((r2 + 128*x0 + 8192*x1) // 32768) + ((((r2 + 128*x0 + 8192*x1) // 64) % 512))), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((r2 + 128*x0 + 8192*x1) // 64) % 512)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (512*((r2 % 64)) + 32768*((r2 + 128*x0 + 8192*x1) // 32768) + ((((r2 + 128*x0 + 8192*x1) // 64) % 512))), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rc/crctp3r4rlvzkx66zqmjc7bummvvaebteu7ybo5glofvrdt6uesp.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   mean => mean
#   sub => sub
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_9, %relu_19), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_17 = async_compile.triton('triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbqsaoonfpjg3gfdxfdou7uzk43gh37hkg4pqyfksxwjhqcofop.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   input_1 => convolution
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => _low_memory_max_pool2d_with_offsets_2
#   input_18 => convolution_7
#   input_19 => relu_7
#   input_2 => relu
#   input_20 => convolution_8
#   input_21 => relu_8
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_3 => convolution_1
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => _low_memory_max_pool2d_with_offsets_4
#   input_34 => convolution_14
#   input_35 => relu_14
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_4 => relu_1
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => convolution_3
#   input_9 => relu_3
#   mean => mean
#   sub => sub
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg21_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_13, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_9, %relu_19), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_18 = async_compile.triton('triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_18(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(arg3_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        buf25 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf1, buf25, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg2_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg2_1
        buf24 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg21_1, buf24, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg21_1
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf28 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(arg3_1, buf4, buf28, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg3_1
        buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf32 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(arg5_1, buf8, buf32, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg5_1
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf35 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(arg7_1, buf11, buf35, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg7_1
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf39 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(arg9_1, buf15, buf39, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg9_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf3, arg1_1, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf3
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf6, arg4_1, 1048576, grid=grid(1048576), stream=stream0)
        buf7 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf6, buf7, 262144, grid=grid(262144), stream=stream0)
        del buf6
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf10, arg6_1, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf10
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf13, arg8_1, 524288, grid=grid(524288), stream=stream0)
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf13, buf14, 131072, grid=grid(131072), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf17, arg10_1, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf24
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf27, arg1_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf27
        del buf28
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf30, arg4_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg4_1
        buf31 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf30, buf31, 262144, grid=grid(262144), stream=stream0)
        del buf30
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf33 = extern_kernels.convolution(buf31, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf31
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf34, arg6_1, 524288, grid=grid(524288), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf34
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf37, arg8_1, 524288, grid=grid(524288), stream=stream0)
        del arg8_1
        buf38 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf37, buf38, 131072, grid=grid(131072), stream=stream0)
        del buf37
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf38
        del buf39
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf41, arg10_1, 262144, grid=grid(262144), stream=stream0)
        del arg10_1
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf42 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg11_1, buf18, buf42, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf17
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf41
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf20, arg12_1, 262144, grid=grid(262144), stream=stream0)
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf44, arg12_1, 262144, grid=grid(262144), stream=stream0)
        del arg12_1
        buf21 = buf42; del buf42  # reuse
        buf45 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg13_1, buf21, buf45, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg13_1
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf20
        del buf21
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf44
        del buf45
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf23, arg14_1, 262144, grid=grid(262144), stream=stream0)
        buf48 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf23, buf48, 65536, grid=grid(65536), stream=stream0)
        del buf23
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf47, arg14_1, 262144, grid=grid(262144), stream=stream0)
        del arg14_1
        buf57 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf47, buf57, 65536, grid=grid(65536), stream=stream0)
        del buf47
        buf49 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf58 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg15_1, buf49, buf58, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg15_1
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf48
        del buf49
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf57
        del buf58
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf51, arg16_1, 131072, grid=grid(131072), stream=stream0)
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf60, arg16_1, 131072, grid=grid(131072), stream=stream0)
        del arg16_1
        buf52 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        buf61 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_41, input_42, input_43], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg17_1, buf52, buf61, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf53 = extern_kernels.convolution(buf51, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf51
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf60
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf54, arg18_1, 131072, grid=grid(131072), stream=stream0)
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf63, arg18_1, 131072, grid=grid(131072), stream=stream0)
        del arg18_1
        buf55 = buf61; del buf61  # reuse
        buf64 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_41, input_42, input_43, input_44, input_45], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(arg19_1, buf55, buf64, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf56 = extern_kernels.convolution(buf54, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf54
        del buf55
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf63
        del buf64
        buf66 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16.run(buf56, arg20_1, buf65, buf66, 1024, 128, grid=grid(1024), stream=stream0)
        del arg20_1
        del buf56
        del buf65
        buf67 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_17.run(buf66, buf67, 16, 64, grid=grid(16), stream=stream0)
        del buf66
        buf68 = empty_strided_cuda((), (), torch.float32)
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_18, input_19, input_20, input_21, input_22, input_23, input_41, input_42, input_43, input_44, input_45, input_46, sub, abs_1, mean], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.sub, aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_18.run(buf69, buf67, 1, 16, grid=grid(1), stream=stream0)
        del buf67
    return (buf69, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
