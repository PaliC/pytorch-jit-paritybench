# AOT ID: ['129_inference']
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
# Topologically Sorted Source Nodes: [input_1, input_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_31 => convolution_13
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg27_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: inductor_cache/5s/c5scyhbqgjw3p5k3uoji7du7b3olgfhia5vwiwytqec447ygidno.py
# Topologically Sorted Source Nodes: [input_3, input_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_33 => convolution_14
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
# Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6, input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: inductor_cache/4b/c4b4qy5kyo5qk4hzp2mp2ok2duwsdxusyggwfbhbvxgpuneeloeq.py
# Topologically Sorted Source Nodes: [input_8, input_38], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_38 => convolution_16
#   input_8 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
# Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_38, input_39, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   %relu : [num_users=9] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool2d_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=9] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_10 => _low_memory_max_pool2d_with_offsets_1
#   input_11 => convolution_4
#   input_12 => relu_4
#   input_8 => convolution_3
#   input_9 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=9] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
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


# kernel path: inductor_cache/yr/cyrctmwrrld527byqpbepouul7bkmwxwlyf7d67m5gnjowebersm.py
# Topologically Sorted Source Nodes: [G_x, G_y, sub, pow_1, mean], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x => div
#   G_y => div_1
#   mean => mean
#   pow_1 => pow_1
#   sub => sub
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 262144), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_1, 262144), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
triton_red_fused_div_mean_pow_sub_11 = async_compile.triton('triton_red_fused_div_mean_pow_sub_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mean_pow_sub_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_mean_pow_sub_11(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 3.814697265625e-06
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgd4xojtcsqsbghndc3aeo47e7hdoklfz5vpweak3l3ybyc3kafc.py
# Topologically Sorted Source Nodes: [G_x_4, G_y_4, sub_4, pow_5, mean_4], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_4 => div_8
#   G_y_4 => div_9
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sub_4 => sub_4
# Graph fragment:
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_8, 131072), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_9, 131072), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_8, %div_9), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_red_fused_div_mean_pow_sub_12 = async_compile.triton('triton_red_fused_div_mean_pow_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mean_pow_sub_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_mean_pow_sub_12(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 7.62939453125e-06
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5c/c5cqwocnxuqmlij47sul4a3vb3stedplbda36gsjbjnmnzdymgif.py
# Topologically Sorted Source Nodes: [G_x_4, G_y_4, sub_4, pow_5, mean_4], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_4 => div_8
#   G_y_4 => div_9
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sub_4 => sub_4
# Graph fragment:
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_8, 131072), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_9, 131072), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_8, %div_9), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_per_fused_div_mean_pow_sub_13 = async_compile.triton('triton_per_fused_div_mean_pow_sub_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mean_pow_sub_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_mean_pow_sub_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/md/cmdvzerui3krhpelrmvkicsksxmb2yjgxdqlikzwdzmcfot27373.py
# Topologically Sorted Source Nodes: [G_x_8, G_y_8, sub_8, pow_9, mean_8], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_8 => div_16
#   G_y_8 => div_17
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sub_8 => sub_8
# Graph fragment:
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_16, 65536), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_17, 65536), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_16, %div_17), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
triton_red_fused_div_mean_pow_sub_14 = async_compile.triton('triton_red_fused_div_mean_pow_sub_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mean_pow_sub_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_mean_pow_sub_14(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1.52587890625e-05
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l6/cl67lknuh6ly3l7l4bsb2m4fgjv35r6numqrjkcwq5f2trotndu7.py
# Topologically Sorted Source Nodes: [G_x_8, G_y_8, sub_8, pow_9, mean_8], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_8 => div_16
#   G_y_8 => div_17
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sub_8 => sub_8
# Graph fragment:
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_16, 65536), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_17, 65536), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_16, %div_17), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
triton_per_fused_div_mean_pow_sub_15 = async_compile.triton('triton_per_fused_div_mean_pow_sub_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mean_pow_sub_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_mean_pow_sub_15(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/h6/ch6wwzb3jbbvh3zp55y6mburejzw5sqc5s4k6w2yoomfnsvar5vd.py
# Topologically Sorted Source Nodes: [input_13, input_43], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_13 => convolution_5
#   input_43 => convolution_18
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/em/cemo5zofillrpg3v2bfejopy23e2jrenzbhvvl2c2336kloupxdd.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => convolution_7
#   input_18 => relu_7
#   input_19 => _low_memory_max_pool2d_with_offsets_2
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_17 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/7q/c7qshnzhewed7kvws7auumrogph6e6jzjfeix7gpnwd7y3ue26q2.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => convolution_7
#   input_18 => relu_7
#   input_19 => _low_memory_max_pool2d_with_offsets_2
#   input_20 => convolution_8
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => convolution_20
#   input_48 => relu_20
#   input_49 => _low_memory_max_pool2d_with_offsets_6
#   input_50 => convolution_21
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_20, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_18 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_18(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/tj/ctjyyofqs3mgfuulvvulcegilpni4yxehqjonf42fmj4zzjn7er4.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_13 => convolution_5
#   input_14 => relu_5
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_17 => convolution_7
#   input_18 => relu_7
#   input_19 => _low_memory_max_pool2d_with_offsets_2
#   input_20 => convolution_8
#   input_21 => relu_8
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=9] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_19 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yh/cyhm7fccrkcg2lakpwamgtilz4lehlvvzj5js64x3grdk4icdjhp.py
# Topologically Sorted Source Nodes: [G_x_12, G_y_12, sub_12, pow_13, mean_12], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_12 => div_24
#   G_y_12 => div_25
#   mean_12 => mean_12
#   pow_13 => pow_13
#   sub_12 => sub_12
# Graph fragment:
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_24, 32768), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_25, 32768), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_24, %div_25), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_13,), kwargs = {})
triton_red_fused_div_mean_pow_sub_20 = async_compile.triton('triton_red_fused_div_mean_pow_sub_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mean_pow_sub_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_mean_pow_sub_20(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 3.0517578125e-05
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpy7nf5jf7lrjojbu252cbvd6i5nwi4oehotfi2elts7r27lzsnr.py
# Topologically Sorted Source Nodes: [input_22, input_52], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_22 => convolution_9
#   input_52 => convolution_22
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xe/cxebvqeenep3ehno56upcqa2ilbfn4gh5zuznuc632tggnorm2n7.py
# Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_22 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 4)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 8192*x2), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 8192*x2), None)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 1024*x1 + 8192*x2), None)
    tmp5 = tl.load(in_ptr0 + (4608 + x0 + 1024*x1 + 8192*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5mdcsfiispsbpsfzld362fmfaicaouv6unhc3fov3msbm476leq.py
# Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_29, input_30, input_59, input_60], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_24 => convolution_10
#   input_25 => relu_10
#   input_26 => convolution_11
#   input_27 => relu_11
#   input_28 => _low_memory_max_pool2d_with_offsets_3
#   input_29 => convolution_12
#   input_30 => relu_12
#   input_52 => convolution_22
#   input_53 => relu_22
#   input_54 => convolution_23
#   input_55 => relu_23
#   input_56 => convolution_24
#   input_57 => relu_24
#   input_58 => _low_memory_max_pool2d_with_offsets_7
#   input_59 => convolution_25
#   input_60 => relu_25
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_24, [2, 2], [2, 2], [0, 0], [1, 1], True), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_23 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 8192*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + 512*x2 + 8192*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp4, xmask)
    tl.store(out_ptr1 + (x2 + 16*y3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f4/cf4zkfhsqd3pet7uetkb6x22p35sz3bziyq2raxyzt2o64n56ek2.py
# Topologically Sorted Source Nodes: [G_x_16, G_y_16, sub_16, pow_17, mean_16], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   G_x_16 => div_32
#   G_y_16 => div_33
#   mean_16 => mean_16
#   pow_17 => pow_17
#   sub_16 => sub_16
# Graph fragment:
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_32, 8192), kwargs = {})
#   %div_33 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_33, 8192), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_32, %div_33), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_16, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_17,), kwargs = {})
triton_red_fused_div_mean_pow_sub_24 = async_compile.triton('triton_red_fused_div_mean_pow_sub_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mean_pow_sub_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_mean_pow_sub_24(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0001220703125
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nu/cnuuldwkom5k34lhpgnvy2c6ag7midpaonz4j5urv47zv5eidsst.py
# Topologically Sorted Source Nodes: [G_x, G_y, sub, pow_1, mean, sqrt, mul, loss, G_x_1, G_y_1, sub_1, pow_2, mean_1, sqrt_1, mul_1, loss_1, G_x_2, G_y_2, sub_2, pow_3, mean_2, sqrt_2, mul_2, loss_2, G_x_3, G_y_3, sub_3, pow_4, mean_3, sqrt_3, mul_3, loss_3, G_x_4, G_y_4, sub_4, pow_5, mean_4, sqrt_4, mul_4, loss_4, G_x_5, G_y_5, sub_5, pow_6, mean_5, sqrt_5, mul_5, loss_5, G_x_6, G_y_6, sub_6, pow_7, mean_6, sqrt_6, mul_6, loss_6, G_x_7, G_y_7, sub_7, pow_8, mean_7, sqrt_7, mul_7, loss_7, G_x_8, G_y_8, sub_8, pow_9, mean_8, sqrt_8, mul_8, loss_8, G_x_9, G_y_9, sub_9, pow_10, mean_9, sqrt_9, mul_9, loss_9, G_x_10, G_y_10, sub_10, pow_11, mean_10, sqrt_10, mul_10, loss_10, G_x_11, G_y_11, sub_11, pow_12, mean_11, sqrt_11, mul_11, loss_11, G_x_12, G_y_12, sub_12, pow_13, mean_12, sqrt_12, mul_12, loss_12, G_x_13, G_y_13, sub_13, pow_14, mean_13, sqrt_13, mul_13, loss_13, G_x_14, G_y_14, sub_14, pow_15, mean_14, sqrt_14, mul_14, loss_14, G_x_15, G_y_15, sub_15, pow_16, mean_15, sqrt_15, mul_15, loss_15, G_x_16, G_y_16, sub_16, pow_17, mean_16, sqrt_16, mul_16, loss_16, G_x_17, G_y_17, sub_17, pow_18, mean_17, sqrt_17, mul_17, loss_17, G_x_18, G_y_18, sub_18, pow_19, mean_18, sqrt_18, mul_18, loss_18, G_x_19, G_y_19, sub_19, pow_20, mean_19, sqrt_19, mul_19, loss_19], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean, aten.sqrt, aten.mul, aten.add]
# Source node to ATen node mapping:
#   G_x => div
#   G_x_1 => div_2
#   G_x_10 => div_20
#   G_x_11 => div_22
#   G_x_12 => div_24
#   G_x_13 => div_26
#   G_x_14 => div_28
#   G_x_15 => div_30
#   G_x_16 => div_32
#   G_x_17 => div_34
#   G_x_18 => div_36
#   G_x_19 => div_38
#   G_x_2 => div_4
#   G_x_3 => div_6
#   G_x_4 => div_8
#   G_x_5 => div_10
#   G_x_6 => div_12
#   G_x_7 => div_14
#   G_x_8 => div_16
#   G_x_9 => div_18
#   G_y => div_1
#   G_y_1 => div_3
#   G_y_10 => div_21
#   G_y_11 => div_23
#   G_y_12 => div_25
#   G_y_13 => div_27
#   G_y_14 => div_29
#   G_y_15 => div_31
#   G_y_16 => div_33
#   G_y_17 => div_35
#   G_y_18 => div_37
#   G_y_19 => div_39
#   G_y_2 => div_5
#   G_y_3 => div_7
#   G_y_4 => div_9
#   G_y_5 => div_11
#   G_y_6 => div_13
#   G_y_7 => div_15
#   G_y_8 => div_17
#   G_y_9 => div_19
#   loss => add
#   loss_1 => add_1
#   loss_10 => add_10
#   loss_11 => add_11
#   loss_12 => add_12
#   loss_13 => add_13
#   loss_14 => add_14
#   loss_15 => add_15
#   loss_16 => add_16
#   loss_17 => add_17
#   loss_18 => add_18
#   loss_19 => add_19
#   loss_2 => add_2
#   loss_3 => add_3
#   loss_4 => add_4
#   loss_5 => add_5
#   loss_6 => add_6
#   loss_7 => add_7
#   loss_8 => add_8
#   loss_9 => add_9
#   mean => mean
#   mean_1 => mean_1
#   mean_10 => mean_10
#   mean_11 => mean_11
#   mean_12 => mean_12
#   mean_13 => mean_13
#   mean_14 => mean_14
#   mean_15 => mean_15
#   mean_16 => mean_16
#   mean_17 => mean_17
#   mean_18 => mean_18
#   mean_19 => mean_19
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mean_4 => mean_4
#   mean_5 => mean_5
#   mean_6 => mean_6
#   mean_7 => mean_7
#   mean_8 => mean_8
#   mean_9 => mean_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   pow_1 => pow_1
#   pow_10 => pow_10
#   pow_11 => pow_11
#   pow_12 => pow_12
#   pow_13 => pow_13
#   pow_14 => pow_14
#   pow_15 => pow_15
#   pow_16 => pow_16
#   pow_17 => pow_17
#   pow_18 => pow_18
#   pow_19 => pow_19
#   pow_2 => pow_2
#   pow_20 => pow_20
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   pow_7 => pow_7
#   pow_8 => pow_8
#   pow_9 => pow_9
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_10 => sqrt_10
#   sqrt_11 => sqrt_11
#   sqrt_12 => sqrt_12
#   sqrt_13 => sqrt_13
#   sqrt_14 => sqrt_14
#   sqrt_15 => sqrt_15
#   sqrt_16 => sqrt_16
#   sqrt_17 => sqrt_17
#   sqrt_18 => sqrt_18
#   sqrt_19 => sqrt_19
#   sqrt_2 => sqrt_2
#   sqrt_3 => sqrt_3
#   sqrt_4 => sqrt_4
#   sqrt_5 => sqrt_5
#   sqrt_6 => sqrt_6
#   sqrt_7 => sqrt_7
#   sqrt_8 => sqrt_8
#   sqrt_9 => sqrt_9
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sub_18 => sub_18
#   sub_19 => sub_19
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 262144), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_1, 262144), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt, 0.03125), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_2, 262144), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_3, 262144), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %div_3), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_1, 0.03125), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_1), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_4, 262144), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_5, 262144), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %div_5), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_2, 0.03125), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_2), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_6, 262144), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_7, 262144), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_6, %div_7), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
#   %sqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_3,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_3, 0.03125), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_3), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_8, 131072), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_9, 131072), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_8, %div_9), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
#   %sqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_4,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_4, 0.0625), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_4), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_10, 131072), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_11, 131072), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %div_11), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_6,), kwargs = {})
#   %sqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_5,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_5, 0.0625), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_5), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_12, 131072), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_13, 131072), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_12, %div_13), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_7,), kwargs = {})
#   %sqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_6,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_6, 0.0625), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_6), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_14, 131072), kwargs = {})
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_15, 131072), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_14, %div_15), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_8,), kwargs = {})
#   %sqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_7,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_7, 0.0625), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_7), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_16, 65536), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_17, 65536), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_16, %div_17), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
#   %sqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_8,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_8, 0.125), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_8), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_18, 65536), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_19, 65536), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_18, %div_19), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_9, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_10,), kwargs = {})
#   %sqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_9,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_9, 0.125), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_9), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_20, 65536), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_21, 65536), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_20, %div_21), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_10, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_11,), kwargs = {})
#   %sqrt_10 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_10,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_10, 0.125), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mul_10), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_22, 65536), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_23, 65536), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_22, %div_23), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_11, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_12,), kwargs = {})
#   %sqrt_11 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_11,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_11, 0.125), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_11), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_24, 32768), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_25, 32768), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_24, %div_25), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_13,), kwargs = {})
#   %sqrt_12 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_12,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_12, 0.25), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_12), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_26, 32768), kwargs = {})
#   %div_27 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_27, 32768), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_26, %div_27), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_13, 2), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_14,), kwargs = {})
#   %sqrt_13 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_13,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_13, 0.25), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_13), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_28, 32768), kwargs = {})
#   %div_29 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_29, 32768), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_28, %div_29), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_14, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_15,), kwargs = {})
#   %sqrt_14 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_14,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_14, 0.25), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mul_14), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_30, 32768), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_31, 32768), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_30, %div_31), kwargs = {})
#   %pow_16 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_15, 2), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_16,), kwargs = {})
#   %sqrt_15 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_15,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_15, 0.25), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %mul_15), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_32, 8192), kwargs = {})
#   %div_33 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_33, 8192), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_32, %div_33), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_16, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_17,), kwargs = {})
#   %sqrt_16 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_16,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_16, 1.0), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %mul_16), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_34, 8192), kwargs = {})
#   %div_35 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_35, 8192), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_34, %div_35), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_17, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_18,), kwargs = {})
#   %sqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_17,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_17, 1.0), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_17), kwargs = {})
#   %div_36 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_36, 8192), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_37, 8192), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_36, %div_37), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_18, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_19,), kwargs = {})
#   %sqrt_18 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_18,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_18, 1.0), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_18), kwargs = {})
#   %div_38 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_38, 8192), kwargs = {})
#   %div_39 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_39, 8192), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_38, %div_39), kwargs = {})
#   %pow_20 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_19, 2), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_20,), kwargs = {})
#   %sqrt_19 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_19,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sqrt_19, 1.0), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %mul_19), kwargs = {})
triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25 = async_compile.triton('triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21), 'tt.equal_to': (20,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 8, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp8 = tl.load(in_ptr2 + (r0), None)
    tmp12 = tl.load(in_ptr3 + (r0), None)
    tmp16 = tl.load(in_ptr4 + (r0), None)
    tmp20 = tl.load(in_ptr5 + (r0), None)
    tmp24 = tl.load(in_ptr6 + (r0), None)
    tmp28 = tl.load(in_ptr7 + (r0), None)
    tmp32 = tl.load(in_out_ptr0 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, 1])
    tmp41 = tl.load(in_ptr8 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, 1])
    tmp47 = tl.load(in_ptr9 + (0))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, 1])
    tmp53 = tl.load(in_ptr10 + (0))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, 1])
    tmp59 = tl.load(in_ptr11 + (0))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 1])
    tmp67 = tl.load(in_ptr12 + (0))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, 1])
    tmp73 = tl.load(in_ptr13 + (0))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, 1])
    tmp79 = tl.load(in_ptr14 + (0))
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK, 1])
    tmp85 = tl.load(in_ptr15 + (0))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, 1])
    tmp93 = tl.load(in_ptr16 + (0))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, 1])
    tmp99 = tl.load(in_ptr17 + (0))
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, 1])
    tmp105 = tl.load(in_ptr18 + (0))
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp34 = 4096.0
    tmp35 = tmp33 / tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = 0.03125
    tmp38 = tmp36 * tmp37
    tmp39 = 0.0
    tmp40 = tmp38 + tmp39
    tmp43 = tmp42 / tmp34
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tmp44 * tmp37
    tmp46 = tmp40 + tmp45
    tmp49 = tmp48 / tmp34
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp50 * tmp37
    tmp52 = tmp46 + tmp51
    tmp55 = tmp54 / tmp34
    tmp56 = libdevice.sqrt(tmp55)
    tmp57 = tmp56 * tmp37
    tmp58 = tmp52 + tmp57
    tmp61 = 16384.0
    tmp62 = tmp60 / tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = 0.0625
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 + tmp65
    tmp69 = tmp68 / tmp61
    tmp70 = libdevice.sqrt(tmp69)
    tmp71 = tmp70 * tmp64
    tmp72 = tmp66 + tmp71
    tmp75 = tmp74 / tmp61
    tmp76 = libdevice.sqrt(tmp75)
    tmp77 = tmp76 * tmp64
    tmp78 = tmp72 + tmp77
    tmp81 = tmp80 / tmp61
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tmp82 * tmp64
    tmp84 = tmp78 + tmp83
    tmp87 = 65536.0
    tmp88 = tmp86 / tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = 0.125
    tmp91 = tmp89 * tmp90
    tmp92 = tmp84 + tmp91
    tmp95 = tmp94 / tmp87
    tmp96 = libdevice.sqrt(tmp95)
    tmp97 = tmp96 * tmp90
    tmp98 = tmp92 + tmp97
    tmp101 = tmp100 / tmp87
    tmp102 = libdevice.sqrt(tmp101)
    tmp103 = tmp102 * tmp90
    tmp104 = tmp98 + tmp103
    tmp107 = tmp106 / tmp87
    tmp108 = libdevice.sqrt(tmp107)
    tmp109 = tmp108 * tmp90
    tmp110 = tmp104 + tmp109
    tmp111 = 262144.0
    tmp112 = tmp3 / tmp111
    tmp113 = libdevice.sqrt(tmp112)
    tmp114 = 0.25
    tmp115 = tmp113 * tmp114
    tmp116 = tmp110 + tmp115
    tmp117 = tmp7 / tmp111
    tmp118 = libdevice.sqrt(tmp117)
    tmp119 = tmp118 * tmp114
    tmp120 = tmp116 + tmp119
    tmp121 = tmp11 / tmp111
    tmp122 = libdevice.sqrt(tmp121)
    tmp123 = tmp122 * tmp114
    tmp124 = tmp120 + tmp123
    tmp125 = tmp15 / tmp111
    tmp126 = libdevice.sqrt(tmp125)
    tmp127 = tmp126 * tmp114
    tmp128 = tmp124 + tmp127
    tmp129 = tmp19 / tmp111
    tmp130 = libdevice.sqrt(tmp129)
    tmp131 = 1.0
    tmp132 = tmp130 * tmp131
    tmp133 = tmp128 + tmp132
    tmp134 = tmp23 / tmp111
    tmp135 = libdevice.sqrt(tmp134)
    tmp136 = tmp135 * tmp131
    tmp137 = tmp133 + tmp136
    tmp138 = tmp31 / tmp111
    tmp139 = libdevice.sqrt(tmp138)
    tmp140 = tmp139 * tmp131
    tmp141 = tmp137 + tmp140
    tmp142 = tmp27 / tmp111
    tmp143 = libdevice.sqrt(tmp142)
    tmp144 = tmp143 * tmp131
    tmp145 = tmp141 + tmp144
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp145, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1 = args
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
    assert_size_stride(arg15_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        buf41 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_31], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf1, buf41, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg2_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg2_1
        buf40 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg27_1, buf40, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg27_1
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf44 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(arg3_1, buf4, buf44, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg3_1
        buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf48 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6, input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(arg5_1, buf8, buf48, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg5_1
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf51 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_38], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(arg7_1, buf11, buf51, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg7_1
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf55 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_38, input_39, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(arg9_1, buf15, buf55, 32768, 9, grid=grid(32768, 9), stream=stream0)
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
        buf80 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (64, 4096), (1, 64), 0), reinterpret_tensor(buf3, (4096, 64), (64, 1), 0), out=buf80)
        buf83 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (64, 4096), (1, 64), 262144), reinterpret_tensor(buf3, (4096, 64), (64, 1), 262144), out=buf83)
        buf86 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (64, 4096), (1, 64), 524288), reinterpret_tensor(buf3, (4096, 64), (64, 1), 524288), out=buf86)
        buf89 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (64, 4096), (1, 64), 786432), reinterpret_tensor(buf3, (4096, 64), (64, 1), 786432), out=buf89)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf3
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf6, arg4_1, 1048576, grid=grid(1048576), stream=stream0)
        buf7 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf6, buf7, 262144, grid=grid(262144), stream=stream0)
        del buf6
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf10, arg6_1, 524288, grid=grid(524288), stream=stream0)
        buf92 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (128, 1024), (1, 128), 0), reinterpret_tensor(buf10, (1024, 128), (128, 1), 0), out=buf92)
        buf96 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (128, 1024), (1, 128), 131072), reinterpret_tensor(buf10, (1024, 128), (128, 1), 131072), out=buf96)
        buf100 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (128, 1024), (1, 128), 262144), reinterpret_tensor(buf10, (1024, 128), (128, 1), 262144), out=buf100)
        buf104 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (128, 1024), (1, 128), 393216), reinterpret_tensor(buf10, (1024, 128), (128, 1), 393216), out=buf104)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf10
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf13, arg8_1, 524288, grid=grid(524288), stream=stream0)
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf13, buf14, 131072, grid=grid(131072), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf17, arg10_1, 262144, grid=grid(262144), stream=stream0)
        buf108 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (256, 256), (1, 256), 0), reinterpret_tensor(buf17, (256, 256), (256, 1), 0), out=buf108)
        buf112 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (256, 256), (1, 256), 65536), reinterpret_tensor(buf17, (256, 256), (256, 1), 65536), out=buf112)
        buf116 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (256, 256), (1, 256), 131072), reinterpret_tensor(buf17, (256, 256), (256, 1), 131072), out=buf116)
        buf120 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (256, 256), (1, 256), 196608), reinterpret_tensor(buf17, (256, 256), (256, 1), 196608), out=buf120)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf40
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf43, arg1_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg1_1
        buf81 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (64, 4096), (1, 64), 0), reinterpret_tensor(buf43, (4096, 64), (64, 1), 0), out=buf81)
        buf82 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x, G_y, sub, pow_1, mean], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_11.run(buf80, buf81, buf82, 1, 4096, grid=grid(1), stream=stream0)
        del buf80
        buf84 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (64, 4096), (1, 64), 262144), reinterpret_tensor(buf43, (4096, 64), (64, 1), 262144), out=buf84)
        buf85 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_1, G_y_1, sub_1, pow_2, mean_1], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_11.run(buf83, buf84, buf85, 1, 4096, grid=grid(1), stream=stream0)
        del buf83
        buf87 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (64, 4096), (1, 64), 524288), reinterpret_tensor(buf43, (4096, 64), (64, 1), 524288), out=buf87)
        buf88 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_2, G_y_2, sub_2, pow_3, mean_2], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_11.run(buf86, buf87, buf88, 1, 4096, grid=grid(1), stream=stream0)
        del buf86
        buf90 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (64, 4096), (1, 64), 786432), reinterpret_tensor(buf43, (4096, 64), (64, 1), 786432), out=buf90)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf43
        del buf44
        buf91 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_3, G_y_3, sub_3, pow_4, mean_3], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_11.run(buf89, buf90, buf91, 1, 4096, grid=grid(1), stream=stream0)
        del buf89
        del buf90
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf46, arg4_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg4_1
        buf47 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34, input_35], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf46, buf47, 262144, grid=grid(262144), stream=stream0)
        del buf46
        # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf49 = extern_kernels.convolution(buf47, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf47
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36, input_37], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf50, arg6_1, 524288, grid=grid(524288), stream=stream0)
        del arg6_1
        buf93 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (128, 1024), (1, 128), 0), reinterpret_tensor(buf50, (1024, 128), (128, 1), 0), out=buf93)
        buf94 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_4, G_y_4, sub_4, pow_5, mean_4], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_12.run(buf92, buf93, buf94, 2, 8192, grid=grid(2), stream=stream0)
        del buf92
        buf95 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_4, G_y_4, sub_4, pow_5, mean_4], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_13.run(buf94, buf95, 1, 2, grid=grid(1), stream=stream0)
        buf97 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (128, 1024), (1, 128), 131072), reinterpret_tensor(buf50, (1024, 128), (128, 1), 131072), out=buf97)
        buf98 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [G_x_5, G_y_5, sub_5, pow_6, mean_5], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_12.run(buf96, buf97, buf98, 2, 8192, grid=grid(2), stream=stream0)
        del buf96
        buf99 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_5, G_y_5, sub_5, pow_6, mean_5], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_13.run(buf98, buf99, 1, 2, grid=grid(1), stream=stream0)
        buf101 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (128, 1024), (1, 128), 262144), reinterpret_tensor(buf50, (1024, 128), (128, 1), 262144), out=buf101)
        buf102 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [G_x_6, G_y_6, sub_6, pow_7, mean_6], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_12.run(buf100, buf101, buf102, 2, 8192, grid=grid(2), stream=stream0)
        del buf100
        buf103 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_6, G_y_6, sub_6, pow_7, mean_6], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_13.run(buf102, buf103, 1, 2, grid=grid(1), stream=stream0)
        buf105 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (128, 1024), (1, 128), 393216), reinterpret_tensor(buf50, (1024, 128), (128, 1), 393216), out=buf105)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf50
        del buf51
        buf106 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [G_x_7, G_y_7, sub_7, pow_8, mean_7], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_12.run(buf104, buf105, buf106, 2, 8192, grid=grid(2), stream=stream0)
        del buf104
        del buf105
        buf107 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_7, G_y_7, sub_7, pow_8, mean_7], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_13.run(buf106, buf107, 1, 2, grid=grid(1), stream=stream0)
        del buf106
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(buf53, arg8_1, 524288, grid=grid(524288), stream=stream0)
        del arg8_1
        buf54 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf53, buf54, 131072, grid=grid(131072), stream=stream0)
        del buf53
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40, input_41], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf56 = extern_kernels.convolution(buf54, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf54
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40, input_41, input_42], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf57, arg10_1, 262144, grid=grid(262144), stream=stream0)
        del arg10_1
        buf109 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (256, 256), (1, 256), 0), reinterpret_tensor(buf57, (256, 256), (256, 1), 0), out=buf109)
        buf110 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_8, G_y_8, sub_8, pow_9, mean_8], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_14.run(buf108, buf109, buf110, 8, 8192, grid=grid(8), stream=stream0)
        del buf108
        buf111 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_8, G_y_8, sub_8, pow_9, mean_8], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_15.run(buf110, buf111, 1, 8, grid=grid(1), stream=stream0)
        buf113 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (256, 256), (1, 256), 65536), reinterpret_tensor(buf57, (256, 256), (256, 1), 65536), out=buf113)
        buf114 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [G_x_9, G_y_9, sub_9, pow_10, mean_9], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_14.run(buf112, buf113, buf114, 8, 8192, grid=grid(8), stream=stream0)
        del buf112
        buf115 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_9, G_y_9, sub_9, pow_10, mean_9], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_15.run(buf114, buf115, 1, 8, grid=grid(1), stream=stream0)
        buf117 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (256, 256), (1, 256), 131072), reinterpret_tensor(buf57, (256, 256), (256, 1), 131072), out=buf117)
        buf118 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [G_x_10, G_y_10, sub_10, pow_11, mean_10], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_14.run(buf116, buf117, buf118, 8, 8192, grid=grid(8), stream=stream0)
        del buf116
        buf119 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_10, G_y_10, sub_10, pow_11, mean_10], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_15.run(buf118, buf119, 1, 8, grid=grid(1), stream=stream0)
        buf121 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (256, 256), (1, 256), 196608), reinterpret_tensor(buf57, (256, 256), (256, 1), 196608), out=buf121)
        buf122 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [G_x_11, G_y_11, sub_11, pow_12, mean_11], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_14.run(buf120, buf121, buf122, 8, 8192, grid=grid(8), stream=stream0)
        buf123 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_11, G_y_11, sub_11, pow_12, mean_11], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mean_pow_sub_15.run(buf122, buf123, 1, 8, grid=grid(1), stream=stream0)
        del buf122
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf58 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_43], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(arg11_1, buf18, buf58, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf17
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf57
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf20, arg12_1, 262144, grid=grid(262144), stream=stream0)
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf60, arg12_1, 262144, grid=grid(262144), stream=stream0)
        del arg12_1
        buf21 = buf58; del buf58  # reuse
        buf61 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_43, input_44, input_45], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(arg13_1, buf21, buf61, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg13_1
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf20
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45], Original ATen: [aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf60
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf23, arg14_1, 262144, grid=grid(262144), stream=stream0)
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf63, arg14_1, 262144, grid=grid(262144), stream=stream0)
        del arg14_1
        buf24 = buf61; del buf61  # reuse
        buf64 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_43, input_44, input_45, input_46, input_47], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(arg15_1, buf24, buf64, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg15_1
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf23
        del buf24
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46, input_47], Original ATen: [aten.convolution, aten.relu]
        buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf64
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf26, arg16_1, 262144, grid=grid(262144), stream=stream0)
        buf27 = reinterpret_tensor(buf121, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf26, buf27, 65536, grid=grid(65536), stream=stream0)
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46, input_47, input_48], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf66, arg16_1, 262144, grid=grid(262144), stream=stream0)
        del arg16_1
        buf67 = reinterpret_tensor(buf120, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46, input_47, input_48, input_49], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf66, buf67, 65536, grid=grid(65536), stream=stream0)
        buf28 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        buf68 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg17_1, buf28, buf68, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf27
        del buf28
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf67
        del buf68
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf30, arg18_1, 131072, grid=grid(131072), stream=stream0)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf70, arg18_1, 131072, grid=grid(131072), stream=stream0)
        del arg18_1
        buf124 = reinterpret_tensor(buf66, (512, 512), (512, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 64), (1, 512), 0), reinterpret_tensor(buf30, (64, 512), (512, 1), 0), out=buf124)
        buf125 = reinterpret_tensor(buf26, (512, 512), (512, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 64), (1, 512), 0), reinterpret_tensor(buf70, (64, 512), (512, 1), 0), out=buf125)
        buf126 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_12, G_y_12, sub_12, pow_13, mean_12], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_20.run(buf124, buf125, buf126, 32, 8192, grid=grid(32), stream=stream0)
        buf128 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 64), (1, 512), 32768), reinterpret_tensor(buf30, (64, 512), (512, 1), 32768), out=buf128)
        buf129 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 64), (1, 512), 32768), reinterpret_tensor(buf70, (64, 512), (512, 1), 32768), out=buf129)
        buf130 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_13, G_y_13, sub_13, pow_14, mean_13], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_20.run(buf128, buf129, buf130, 32, 8192, grid=grid(32), stream=stream0)
        buf132 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 64), (1, 512), 65536), reinterpret_tensor(buf30, (64, 512), (512, 1), 65536), out=buf132)
        buf133 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 64), (1, 512), 65536), reinterpret_tensor(buf70, (64, 512), (512, 1), 65536), out=buf133)
        buf134 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_14, G_y_14, sub_14, pow_15, mean_14], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_20.run(buf132, buf133, buf134, 32, 8192, grid=grid(32), stream=stream0)
        buf136 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 64), (1, 512), 98304), reinterpret_tensor(buf30, (64, 512), (512, 1), 98304), out=buf136)
        buf137 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 64), (1, 512), 98304), reinterpret_tensor(buf70, (64, 512), (512, 1), 98304), out=buf137)
        buf138 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_15, G_y_15, sub_15, pow_16, mean_15], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_20.run(buf136, buf137, buf138, 32, 8192, grid=grid(32), stream=stream0)
        buf31 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        buf71 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_52], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(arg19_1, buf31, buf71, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf30
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf70
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf33, arg20_1, 131072, grid=grid(131072), stream=stream0)
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf73, arg20_1, 131072, grid=grid(131072), stream=stream0)
        del arg20_1
        buf34 = buf71; del buf71  # reuse
        buf74 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_52, input_53, input_54], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(arg21_1, buf34, buf74, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf33
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54], Original ATen: [aten.convolution, aten.relu]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf73
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf36, arg22_1, 131072, grid=grid(131072), stream=stream0)
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf76, arg22_1, 131072, grid=grid(131072), stream=stream0)
        del arg22_1
        buf37 = buf74; del buf74  # reuse
        buf77 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_52, input_53, input_54, input_55, input_56], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(arg23_1, buf37, buf77, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26], Original ATen: [aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf36
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56], Original ATen: [aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf76
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf39, arg24_1, 131072, grid=grid(131072), stream=stream0)
        buf140 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_22.run(buf39, buf140, 32768, grid=grid(32768), stream=stream0)
        del buf39
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56, input_57], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf79, arg24_1, 131072, grid=grid(131072), stream=stream0)
        del arg24_1
        buf145 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56, input_57, input_58], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_22.run(buf79, buf145, 32768, grid=grid(32768), stream=stream0)
        del buf79
        buf141 = buf77; del buf77  # reuse
        buf146 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_29, input_59], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(arg25_1, buf141, buf146, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf142 = extern_kernels.convolution(buf140, buf141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf141
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf146
        buf143 = reinterpret_tensor(buf145, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf145  # reuse
        buf148 = reinterpret_tensor(buf140, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_29, input_30, input_59, input_60], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_23.run(buf142, arg26_1, buf147, buf143, buf148, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del arg26_1
        del buf142
        del buf147
        buf144 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 16), (16, 1), 0), reinterpret_tensor(buf143, (16, 512), (1, 16), 0), out=buf144)
        buf149 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 16), (16, 1), 0), reinterpret_tensor(buf148, (16, 512), (1, 16), 0), out=buf149)
        buf150 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_16, G_y_16, sub_16, pow_17, mean_16], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_24.run(buf144, buf149, buf150, 32, 8192, grid=grid(32), stream=stream0)
        buf152 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 16), (16, 1), 8192), reinterpret_tensor(buf143, (16, 512), (1, 16), 8192), out=buf152)
        buf153 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 16), (16, 1), 8192), reinterpret_tensor(buf148, (16, 512), (1, 16), 8192), out=buf153)
        buf154 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_17, G_y_17, sub_17, pow_18, mean_17], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_24.run(buf152, buf153, buf154, 32, 8192, grid=grid(32), stream=stream0)
        buf156 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 16), (16, 1), 16384), reinterpret_tensor(buf143, (16, 512), (1, 16), 16384), out=buf156)
        buf161 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 16), (16, 1), 24576), reinterpret_tensor(buf143, (16, 512), (1, 16), 24576), out=buf161)
        del buf143
        buf157 = reinterpret_tensor(buf63, (512, 512), (512, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 16), (16, 1), 16384), reinterpret_tensor(buf148, (16, 512), (1, 16), 16384), out=buf157)
        buf158 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_18, G_y_18, sub_18, pow_19, mean_18], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_24.run(buf156, buf157, buf158, 32, 8192, grid=grid(32), stream=stream0)
        del buf156
        buf162 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 16), (16, 1), 24576), reinterpret_tensor(buf148, (16, 512), (1, 16), 24576), out=buf162)
        del buf148
        buf163 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [G_x_19, G_y_19, sub_19, pow_20, mean_19], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_mean_pow_sub_24.run(buf161, buf162, buf163, 32, 8192, grid=grid(32), stream=stream0)
        del buf161
        del buf162
        buf160 = buf82; del buf82  # reuse
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [G_x, G_y, sub, pow_1, mean, sqrt, mul, loss, G_x_1, G_y_1, sub_1, pow_2, mean_1, sqrt_1, mul_1, loss_1, G_x_2, G_y_2, sub_2, pow_3, mean_2, sqrt_2, mul_2, loss_2, G_x_3, G_y_3, sub_3, pow_4, mean_3, sqrt_3, mul_3, loss_3, G_x_4, G_y_4, sub_4, pow_5, mean_4, sqrt_4, mul_4, loss_4, G_x_5, G_y_5, sub_5, pow_6, mean_5, sqrt_5, mul_5, loss_5, G_x_6, G_y_6, sub_6, pow_7, mean_6, sqrt_6, mul_6, loss_6, G_x_7, G_y_7, sub_7, pow_8, mean_7, sqrt_7, mul_7, loss_7, G_x_8, G_y_8, sub_8, pow_9, mean_8, sqrt_8, mul_8, loss_8, G_x_9, G_y_9, sub_9, pow_10, mean_9, sqrt_9, mul_9, loss_9, G_x_10, G_y_10, sub_10, pow_11, mean_10, sqrt_10, mul_10, loss_10, G_x_11, G_y_11, sub_11, pow_12, mean_11, sqrt_11, mul_11, loss_11, G_x_12, G_y_12, sub_12, pow_13, mean_12, sqrt_12, mul_12, loss_12, G_x_13, G_y_13, sub_13, pow_14, mean_13, sqrt_13, mul_13, loss_13, G_x_14, G_y_14, sub_14, pow_15, mean_14, sqrt_14, mul_14, loss_14, G_x_15, G_y_15, sub_15, pow_16, mean_15, sqrt_15, mul_15, loss_15, G_x_16, G_y_16, sub_16, pow_17, mean_16, sqrt_16, mul_16, loss_16, G_x_17, G_y_17, sub_17, pow_18, mean_17, sqrt_17, mul_17, loss_17, G_x_18, G_y_18, sub_18, pow_19, mean_18, sqrt_18, mul_18, loss_18, G_x_19, G_y_19, sub_19, pow_20, mean_19, sqrt_19, mul_19, loss_19], Original ATen: [aten.div, aten.sub, aten.pow, aten.mean, aten.sqrt, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25.run(buf165, buf126, buf130, buf134, buf138, buf150, buf154, buf163, buf158, buf85, buf88, buf91, buf95, buf99, buf103, buf107, buf111, buf115, buf119, buf123, 1, 32, grid=grid(1), stream=stream0)
        del buf103
        del buf107
        del buf111
        del buf115
        del buf119
        del buf123
        del buf126
        del buf130
        del buf134
        del buf138
        del buf150
        del buf154
        del buf158
        del buf163
        del buf85
        del buf88
        del buf91
        del buf95
        del buf99
    return (buf165, )


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
    arg15_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
