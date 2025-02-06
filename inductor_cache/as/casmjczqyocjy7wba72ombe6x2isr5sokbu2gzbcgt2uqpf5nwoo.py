# AOT ID: ['8_inference']
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


# kernel path: inductor_cache/yg/cygiloplcagrd62cgscfdgf7wklh4324n2v5x4jjzpiu7n6zevpe.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/4w/c4w56uj623rv7utt6rmgwegxh5z2bc2yrqic6doh4cnxdnqsk47f.py
# Topologically Sorted Source Nodes: [x, relu, x_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   relu => relu
#   x => convolution
#   x_1 => convolution_1
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/xy/cxyn62u2iqupze63h34ewh3l2nz2zz5llxgr52mynnejk25eiihh.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/fc/cfc2cmdjajknu2sitiuh5rxvjgz4slxjliqssqylcd6t6gnxzlzl.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/tt/cttf7i4ma7sl5i67kfdbbk7x7zwucvpmj7ffbsbyfp24qqrdw5dv.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxv64zfuzsmpmcl4lde2wx5pafft3ls4yj72vd26ufbazaqg2od2.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/ad/cad6bqx2sa23wp5khrnmzn4av27hdqb4lexz762r76vvv46gzbej.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/xx/cxxfkvmzp6wvbrmgpthrrmlzd263grm7yj6v347vclm3wvm3qlbm.py
# Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   loss => mean
#   pow_1 => pow_1
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
triton_per_fused_mean_pow_sub_7 = async_compile.triton('triton_per_fused_mean_pow_sub_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_7(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((r2 % 64)) + 4096*((((r2 + 128*x1) // 64) % 64)) + 262144*((r2 + 128*x1 + 8192*x0) // 262144) + ((((r2 + 128*x1 + 8192*x0) // 4096) % 64))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((r2 + 128*x1 + 8192*x0) // 4096) % 64)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (524288 + 64*((r2 % 64)) + 4096*((((r2 + 128*x1) // 64) % 64)) + 262144*((r2 + 128*x1 + 8192*x0) // 262144) + ((((r2 + 128*x1 + 8192*x0) // 4096) % 64))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/i2/ci2xwbho4od4g7iuqycdxsgmmxkuol3po3i4sgn2czdxcnuowh3h.py
# Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   loss => mean
#   pow_1 => pow_1
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
triton_per_fused_mean_pow_sub_8 = async_compile.triton('triton_per_fused_mean_pow_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7c626wnv5fbubg2don5w5nklzjp36ipwxyhsth7gipw62s4nve.py
# Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   loss => mean
#   pow_1 => pow_1
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
triton_per_fused_mean_pow_sub_9 = async_compile.triton('triton_per_fused_mean_pow_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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


# kernel path: inductor_cache/gn/cgnydbg5n2qzf752mfwpmbh7u3dkang2y6xt76hhv6cilkzcg6nd.py
# Topologically Sorted Source Nodes: [x, relu], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   relu => relu
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_poi_fused_convolution_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/v2/cv2ipejqkcjydj6rdoxzm2x47zti3ybasja6qtsy5ygi4vaxcz6j.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5r/c5rbpenr2izbfijc3owxy5dzs5og2o4gpvbi6evi2eqyxorlv3yn.py
# Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   pow_3 => pow_3
#   sub_2 => sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_5, %slice_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
triton_per_fused_mean_pow_sub_12 = async_compile.triton('triton_per_fused_mean_pow_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_12(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (128*((r2 % 32)) + 4096*((((r2 + 128*x1) // 32) % 32)) + 131072*((r2 + 128*x1 + 8192*x0) // 131072) + ((((r2 + 128*x1 + 8192*x0) // 1024) % 128))), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((r2 + 128*x1 + 8192*x0) // 1024) % 128)), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (262144 + 128*((r2 % 32)) + 4096*((((r2 + 128*x1) // 32) % 32)) + 131072*((r2 + 128*x1 + 8192*x0) // 131072) + ((((r2 + 128*x1 + 8192*x0) // 1024) % 128))), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ls/clsdhhh3ssebliamu6jda4rygwf3vrbxlbpvll3xxbort2q3l6hw.py
# Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   pow_3 => pow_3
#   sub_2 => sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_5, %slice_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
triton_per_fused_mean_pow_sub_13 = async_compile.triton('triton_per_fused_mean_pow_sub_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fc/cfc2c2fz7kjs5thx7cocrsfgrmvgtxms6qnbgbirfscmrymrlzxm.py
# Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   pow_3 => pow_3
#   sub_2 => sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_5, %slice_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
triton_per_fused_mean_pow_sub_14 = async_compile.triton('triton_per_fused_mean_pow_sub_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/sp/cspqyoiiqlvo47t2hchrdbte5ccfgwkdgh26ok7y7jrrvtmz3axg.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/va/cva2uofzijmobkjayit6e26yghhgkgyxfobqhw62oi6hroc2wtux.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_16 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/b7/cb7zp2htq3yapr37cpdirczttpokve7uwyjrfgggem5trs3gbhiu.py
# Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sub_4 => sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_9, %slice_10), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_per_fused_mean_pow_sub_17 = async_compile.triton('triton_per_fused_mean_pow_sub_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_17(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256*((r2 % 16)) + 4096*((((r2 + 128*x1) // 16) % 16)) + 65536*((r2 + 128*x1 + 8192*x0) // 65536) + ((((r2 + 128*x1 + 8192*x0) // 256) % 256))), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((r2 + 128*x1 + 8192*x0) // 256) % 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (131072 + 256*((r2 % 16)) + 4096*((((r2 + 128*x1) // 16) % 16)) + 65536*((r2 + 128*x1 + 8192*x0) // 65536) + ((((r2 + 128*x1 + 8192*x0) // 256) % 256))), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3o/c3ohdlquhedl5dltshix4a5dzbsipdqcpxbvqophhkv2atvdkfri.py
# Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sub_4 => sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_9, %slice_10), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_per_fused_mean_pow_sub_18 = async_compile.triton('triton_per_fused_mean_pow_sub_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4b/c4bdio4olwwveds63zicyioysodntdaddvarh2clgxacn3tfgxjs.py
# Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sub_4 => sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_9, %slice_10), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
triton_per_fused_mean_pow_sub_19 = async_compile.triton('triton_per_fused_mean_pow_sub_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_19(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/nm/cnm6s6agkkdyhfddh566bvwfifmsq3yx5gjvknbrjzjmtp2qbuyu.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   x => convolution
#   x_1 => convolution_1
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_20 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/rl/crlupn5imp7grii42lp55rruaow7ode6aniccbyv6vdnawmowpyz.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_21 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gp/cgpjhghceecvnxjyxqevydfy6xh4x44gmlvxbgpuord5pga7t56d.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_11 => convolution_8
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_22 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/cj/ccj2bfphtlqytgmy6peisyxga2w52nowpwdwmy3g3tkh2oy5pgme.py
# Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sub_8 => sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_17, %slice_18), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
triton_per_fused_mean_pow_sub_23 = async_compile.triton('triton_per_fused_mean_pow_sub_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_23(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp3 = tl.load(in_ptr0 + (65536 + 512*((r2 % 64)) + 32768*((r2 + 128*x0 + 8192*x1) // 32768) + ((((r2 + 128*x0 + 8192*x1) // 64) % 512))), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4d/c4drjbtfx4ecita3euhafwb422mqncssfvhcxctoabtqnbp7omle.py
# Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sub_8 => sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_17, %slice_18), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
triton_per_fused_mean_pow_sub_24 = async_compile.triton('triton_per_fused_mean_pow_sub_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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


# kernel path: inductor_cache/v2/cv24vqnpa4ytssvaqrcgke7jlreaiwsah7bpl7b4bmqhtdhabdsv.py
# Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sub_8 => sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_17, %slice_18), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
triton_per_fused_mean_pow_sub_25 = async_compile.triton('triton_per_fused_mean_pow_sub_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_25(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/e4/ce4dx6zqfbszvd26u4ibalo6zumomdrfod25twxnkuhkpxylxz3x.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   relu_8 => relu_8
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_11 => convolution_8
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_26 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/hx/chxawdal7scydlriy5u6nfyhojfs5llq62525fqpdildpfqe7qhx.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   relu_8 => relu_8
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_11 => convolution_8
#   x_12 => convolution_9
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_27 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxcg3w6yi73vl2blzqwacesv2fmufrk7n2r5tju7g6biwwrrq6j.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_10 => relu_10
#   relu_11 => relu_11
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   relu_8 => relu_8
#   relu_9 => relu_9
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_11 => convolution_8
#   x_12 => convolution_9
#   x_13 => convolution_10
#   x_14 => convolution_11
#   x_15 => _low_memory_max_pool2d_with_offsets_3
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_28 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/co/ccogwlezhowuk6po6yzfpiritqemz3x5h7wa55psroiavpydgt7e.py
# Topologically Sorted Source Nodes: [sub_12, pow_13, mean_12], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_12 => mean_12
#   pow_13 => pow_13
#   sub_12 => sub_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_25, %slice_26), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_13,), kwargs = {})
triton_per_fused_mean_pow_sub_29 = async_compile.triton('triton_per_fused_mean_pow_sub_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_29(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (8*x0 + 512*((r2 % 16)) + 8192*x1 + 8192*((r2 + 128*x0) // 8192) + (r2 // 16)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (8*x0 + (r2 // 16)), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (16384 + 8*x0 + 512*((r2 % 16)) + 8192*x1 + 8192*((r2 + 128*x0) // 8192) + (r2 // 16)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sh/cshbkhg5tw5q2wt5adowog37rmorwg2q3miprcdg64jvejr7gxgp.py
# Topologically Sorted Source Nodes: [sub_12, pow_13, mean_12], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_12 => mean_12
#   pow_13 => pow_13
#   sub_12 => sub_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_25, %slice_26), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_13,), kwargs = {})
triton_per_fused_mean_pow_sub_30 = async_compile.triton('triton_per_fused_mean_pow_sub_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_sub_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_pow_sub_30(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
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


# kernel path: inductor_cache/a2/ca2umwfrkuxvyshuvdfivadssi4iawwlbuzugiwwvbj5cswuxlfz.py
# Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   relu => relu
#   relu_1 => relu_1
#   relu_10 => relu_10
#   relu_11 => relu_11
#   relu_12 => relu_12
#   relu_2 => relu_2
#   relu_3 => relu_3
#   relu_4 => relu_4
#   relu_5 => relu_5
#   relu_6 => relu_6
#   relu_7 => relu_7
#   relu_8 => relu_8
#   relu_9 => relu_9
#   x => convolution
#   x_1 => convolution_1
#   x_10 => _low_memory_max_pool2d_with_offsets_2
#   x_11 => convolution_8
#   x_12 => convolution_9
#   x_13 => convolution_10
#   x_14 => convolution_11
#   x_15 => _low_memory_max_pool2d_with_offsets_3
#   x_16 => convolution_12
#   x_2 => _low_memory_max_pool2d_with_offsets
#   x_3 => convolution_2
#   x_4 => convolution_3
#   x_5 => _low_memory_max_pool2d_with_offsets_1
#   x_6 => convolution_4
#   x_7 => convolution_5
#   x_8 => convolution_6
#   x_9 => convolution_7
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg9_1, %arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg11_1, %arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
#   %convolution_6 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg15_1, %arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg17_1, %arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_8,), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg21_1, %arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
#   %convolution_11 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_11, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_12 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_31 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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


# kernel path: inductor_cache/2a/c2agkq3ralhccjtctdirssv2iftxmjdhky3uaclbdz2ddnwzaxyz.py
# Topologically Sorted Source Nodes: [sub, pow_1, loss, sub_1, pow_2, mean_1, loss_1, sub_2, pow_3, mean_2, loss_2, sub_3, pow_4, mean_3, loss_3, sub_4, pow_5, mean_4, loss_4, sub_5, pow_6, mean_5, loss_5, sub_6, pow_7, mean_6, loss_6, sub_7, pow_8, mean_7, loss_7, sub_8, pow_9, mean_8, loss_8, sub_9, pow_10, mean_9, loss_9, sub_10, pow_11, mean_10, loss_10, sub_11, pow_12, mean_11, loss_11, sub_12, pow_13, mean_12, loss_12, sub_13, pow_14, mean_13, loss_13, sub_14, pow_15, mean_14, loss_14, sub_15, pow_16, mean_15, loss_15, truediv], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.div]
# Source node to ATen node mapping:
#   loss => mean
#   loss_1 => add
#   loss_10 => add_9
#   loss_11 => add_10
#   loss_12 => add_11
#   loss_13 => add_12
#   loss_14 => add_13
#   loss_15 => add_14
#   loss_2 => add_1
#   loss_3 => add_2
#   loss_4 => add_3
#   loss_5 => add_4
#   loss_6 => add_5
#   loss_7 => add_6
#   loss_8 => add_7
#   loss_9 => add_8
#   mean_1 => mean_1
#   mean_10 => mean_10
#   mean_11 => mean_11
#   mean_12 => mean_12
#   mean_13 => mean_13
#   mean_14 => mean_14
#   mean_15 => mean_15
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mean_4 => mean_4
#   mean_5 => mean_5
#   mean_6 => mean_6
#   mean_7 => mean_7
#   mean_8 => mean_8
#   mean_9 => mean_9
#   pow_1 => pow_1
#   pow_10 => pow_10
#   pow_11 => pow_11
#   pow_12 => pow_12
#   pow_13 => pow_13
#   pow_14 => pow_14
#   pow_15 => pow_15
#   pow_16 => pow_16
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   pow_7 => pow_7
#   pow_8 => pow_8
#   pow_9 => pow_9
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   truediv => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_4), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_5, %slice_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mean_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_7, %slice_8), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mean_3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_9, %slice_10), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_5,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mean_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_11, %slice_12), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_6,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mean_5), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_13, %slice_14), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_7,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mean_6), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_15, %slice_16), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_8,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mean_7), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_17, %slice_18), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_8, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_9,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mean_8), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_19, %slice_20), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_9, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_10,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mean_9), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_21, %slice_22), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_10, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_11,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mean_10), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_23, %slice_24), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_11, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_12,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mean_11), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_25, %slice_26), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_13,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mean_12), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_27, %slice_28), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_13, 2), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_14,), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mean_13), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_29, %slice_30), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_14, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_15,), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mean_14), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_31, %slice_32), kwargs = {})
#   %pow_16 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_15, 2), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_16,), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mean_15), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_14, 16), kwargs = {})
triton_per_fused_add_div_mean_pow_sub_32 = async_compile.triton('triton_per_fused_add_div_mean_pow_sub_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': (16,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_pow_sub_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_pow_sub_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp8 = tl.load(in_ptr2 + (r0), None)
    tmp12 = tl.load(in_ptr3 + (r0), None)
    tmp16 = tl.load(in_out_ptr0 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, 1])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, 1])
    tmp24 = tl.load(in_ptr5 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, 1])
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp33 = tl.load(in_ptr7 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp38 = tl.load(in_ptr8 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, 1])
    tmp42 = tl.load(in_ptr9 + (0))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, 1])
    tmp46 = tl.load(in_ptr10 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, 1])
    tmp50 = tl.load(in_ptr11 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, 1])
    tmp55 = tl.load(in_ptr12 + (0))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, 1])
    tmp59 = tl.load(in_ptr13 + (0))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 1])
    tmp63 = tl.load(in_ptr14 + (0))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp18 = 524288.0
    tmp19 = tmp17 / tmp18
    tmp22 = tmp21 / tmp18
    tmp23 = tmp19 + tmp22
    tmp26 = 262144.0
    tmp27 = tmp25 / tmp26
    tmp28 = tmp23 + tmp27
    tmp31 = tmp30 / tmp26
    tmp32 = tmp28 + tmp31
    tmp35 = 131072.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp32 + tmp36
    tmp40 = tmp39 / tmp35
    tmp41 = tmp37 + tmp40
    tmp44 = tmp43 / tmp35
    tmp45 = tmp41 + tmp44
    tmp48 = tmp47 / tmp35
    tmp49 = tmp45 + tmp48
    tmp52 = 65536.0
    tmp53 = tmp51 / tmp52
    tmp54 = tmp49 + tmp53
    tmp57 = tmp56 / tmp52
    tmp58 = tmp54 + tmp57
    tmp61 = tmp60 / tmp52
    tmp62 = tmp58 + tmp61
    tmp65 = tmp64 / tmp52
    tmp66 = tmp62 + tmp65
    tmp67 = 16384.0
    tmp68 = tmp3 / tmp67
    tmp69 = tmp66 + tmp68
    tmp70 = tmp7 / tmp67
    tmp71 = tmp69 + tmp70
    tmp72 = tmp11 / tmp67
    tmp73 = tmp71 + tmp72
    tmp74 = tmp15 / tmp67
    tmp75 = tmp73 + tmp74
    tmp76 = 0.0625
    tmp77 = tmp75 * tmp76
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp77, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
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
    assert_size_stride(arg27_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg32_1, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(arg3_1, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg3_1
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(arg2_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del arg2_1
        buf8 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(arg5_1, buf8, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg5_1
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(arg7_1, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg7_1
        buf15 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(arg9_1, buf15, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg9_1
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(arg11_1, buf18, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg11_1
        buf21 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(arg13_1, buf21, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg13_1
        buf24 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(arg15_1, buf24, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg15_1
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf0
        del buf1
        buf40 = empty_strided_cuda((64, 64), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_7.run(buf2, arg1_1, buf40, 4096, 128, grid=grid(4096), stream=stream0)
        buf41 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_8.run(buf40, buf41, 64, 64, grid=grid(64), stream=stream0)
        buf42 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub, pow_1, loss], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_9.run(buf41, buf42, 1, 64, grid=grid(1), stream=stream0)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x, relu], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf3, arg1_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [x, relu, x_1], Original ATen: [aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 64, 64), (262144, 1, 4096, 64))
        del buf3
        del buf4
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [sub_1, pow_2, mean_1], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_7.run(buf5, arg4_1, buf43, 4096, 128, grid=grid(4096), stream=stream0)
        buf44 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [sub_1, pow_2, mean_1], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_8.run(buf43, buf44, 64, 64, grid=grid(64), stream=stream0)
        del buf43
        buf45 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, pow_2, mean_1], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_9.run(buf44, buf45, 1, 64, grid=grid(1), stream=stream0)
        del buf44
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf6, arg4_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg4_1
        buf7 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(buf6, buf7, 262144, grid=grid(262144), stream=stream0)
        del buf6
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf7
        del buf8
        buf46 = empty_strided_cuda((32, 64), (1, 32), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_12.run(buf9, arg6_1, buf46, 2048, 128, grid=grid(2048), stream=stream0)
        buf47 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_13.run(buf46, buf47, 32, 64, grid=grid(32), stream=stream0)
        buf48 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, pow_3, mean_2], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_14.run(buf47, buf48, 1, 32, grid=grid(1), stream=stream0)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(buf10, arg6_1, 524288, grid=grid(524288), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 32, 32), (131072, 1, 4096, 128))
        del buf10
        del buf11
        buf49 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [sub_3, pow_4, mean_3], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_12.run(buf12, arg8_1, buf49, 2048, 128, grid=grid(2048), stream=stream0)
        buf50 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [sub_3, pow_4, mean_3], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_13.run(buf49, buf50, 32, 64, grid=grid(32), stream=stream0)
        del buf49
        buf51 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_3, pow_4, mean_3], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_14.run(buf50, buf51, 1, 32, grid=grid(1), stream=stream0)
        del buf50
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(buf13, arg8_1, 524288, grid=grid(524288), stream=stream0)
        del arg8_1
        buf14 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_16.run(buf13, buf14, 131072, grid=grid(131072), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf14
        del buf15
        buf52 = empty_strided_cuda((16, 64), (1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_17.run(buf16, arg10_1, buf52, 1024, 128, grid=grid(1024), stream=stream0)
        buf53 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_18.run(buf52, buf53, 16, 64, grid=grid(16), stream=stream0)
        buf54 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_4, pow_5, mean_4], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_19.run(buf53, buf54, 1, 16, grid=grid(1), stream=stream0)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf17, arg10_1, 262144, grid=grid(262144), stream=stream0)
        del arg10_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf17
        del buf18
        buf55 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [sub_5, pow_6, mean_5], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_17.run(buf19, arg12_1, buf55, 1024, 128, grid=grid(1024), stream=stream0)
        buf56 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [sub_5, pow_6, mean_5], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_18.run(buf55, buf56, 16, 64, grid=grid(16), stream=stream0)
        buf57 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_5, pow_6, mean_5], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_19.run(buf56, buf57, 1, 16, grid=grid(1), stream=stream0)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf20, arg12_1, 262144, grid=grid(262144), stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf20
        del buf21
        buf58 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [sub_6, pow_7, mean_6], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_17.run(buf22, arg14_1, buf58, 1024, 128, grid=grid(1024), stream=stream0)
        buf59 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [sub_6, pow_7, mean_6], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_18.run(buf58, buf59, 16, 64, grid=grid(16), stream=stream0)
        buf60 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_6, pow_7, mean_6], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_19.run(buf59, buf60, 1, 16, grid=grid(1), stream=stream0)
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf23, arg14_1, 262144, grid=grid(262144), stream=stream0)
        del arg14_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 256, 16, 16), (65536, 1, 4096, 256))
        del buf23
        del buf24
        buf61 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [sub_7, pow_8, mean_7], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_17.run(buf25, arg16_1, buf61, 1024, 128, grid=grid(1024), stream=stream0)
        buf62 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sub_7, pow_8, mean_7], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_18.run(buf61, buf62, 16, 64, grid=grid(16), stream=stream0)
        del buf61
        buf63 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_7, pow_8, mean_7], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_19.run(buf62, buf63, 1, 16, grid=grid(1), stream=stream0)
        del buf62
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf26, arg16_1, 262144, grid=grid(262144), stream=stream0)
        del arg16_1
        buf27 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_21.run(buf26, buf27, 65536, grid=grid(65536), stream=stream0)
        del buf26
        buf28 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_22.run(arg17_1, buf28, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf27
        del buf28
        buf64 = empty_strided_cuda((8, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_23.run(buf29, arg18_1, buf64, 512, 128, grid=grid(512), stream=stream0)
        buf65 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_24.run(buf64, buf65, 8, 64, grid=grid(8), stream=stream0)
        buf66 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_8, pow_9, mean_8], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_25.run(buf65, buf66, 1, 8, grid=grid(1), stream=stream0)
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_26.run(buf30, arg18_1, 131072, grid=grid(131072), stream=stream0)
        del arg18_1
        buf31 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg19_1, buf31, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg19_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf30
        buf67 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [sub_9, pow_10, mean_9], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_23.run(buf32, arg20_1, buf67, 512, 128, grid=grid(512), stream=stream0)
        buf68 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [sub_9, pow_10, mean_9], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_24.run(buf67, buf68, 8, 64, grid=grid(8), stream=stream0)
        buf69 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_9, pow_10, mean_9], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_25.run(buf68, buf69, 1, 8, grid=grid(1), stream=stream0)
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_26.run(buf33, arg20_1, 131072, grid=grid(131072), stream=stream0)
        del arg20_1
        buf34 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg21_1, buf34, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf33
        buf70 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [sub_10, pow_11, mean_10], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_23.run(buf35, arg22_1, buf70, 512, 128, grid=grid(512), stream=stream0)
        buf71 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [sub_10, pow_11, mean_10], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_24.run(buf70, buf71, 8, 64, grid=grid(8), stream=stream0)
        buf72 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_10, pow_11, mean_10], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_25.run(buf71, buf72, 1, 8, grid=grid(1), stream=stream0)
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_26.run(buf36, arg22_1, 131072, grid=grid(131072), stream=stream0)
        del arg22_1
        buf37 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg23_1, buf37, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 8, 8), (32768, 1, 4096, 512))
        del buf36
        buf73 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [sub_11, pow_12, mean_11], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_23.run(buf38, arg24_1, buf73, 512, 128, grid=grid(512), stream=stream0)
        buf74 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [sub_11, pow_12, mean_11], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_24.run(buf73, buf74, 8, 64, grid=grid(8), stream=stream0)
        del buf73
        buf75 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_11, pow_12, mean_11], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_25.run(buf74, buf75, 1, 8, grid=grid(1), stream=stream0)
        del buf74
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_26.run(buf39, arg24_1, 131072, grid=grid(131072), stream=stream0)
        del arg24_1
        buf76 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_28.run(buf39, buf76, 32768, grid=grid(32768), stream=stream0)
        del buf39
        buf77 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg25_1, buf77, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf76
        buf79 = empty_strided_cuda((2, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub_12, pow_13, mean_12], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_29.run(buf78, arg26_1, buf79, 128, 128, grid=grid(128), stream=stream0)
        buf80 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_12, pow_13, mean_12], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_30.run(buf79, buf80, 2, 64, grid=grid(2), stream=stream0)
        buf82 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_31.run(buf82, arg26_1, 32768, grid=grid(32768), stream=stream0)
        del arg26_1
        buf83 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg27_1, buf83, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf82
        buf85 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sub_13, pow_14, mean_13], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_29.run(buf84, arg28_1, buf85, 128, 128, grid=grid(128), stream=stream0)
        buf86 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_13, pow_14, mean_13], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_30.run(buf85, buf86, 2, 64, grid=grid(2), stream=stream0)
        buf88 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_31.run(buf88, arg28_1, 32768, grid=grid(32768), stream=stream0)
        del arg28_1
        buf89 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13, x_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg29_1, buf89, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13, x_18], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf88
        buf91 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [sub_14, pow_15, mean_14], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_29.run(buf90, arg30_1, buf91, 128, 128, grid=grid(128), stream=stream0)
        buf92 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_14, pow_15, mean_14], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_30.run(buf91, buf92, 2, 64, grid=grid(2), stream=stream0)
        buf94 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13, x_18, relu_14], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_31.run(buf94, arg30_1, 32768, grid=grid(32768), stream=stream0)
        del arg30_1
        buf95 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13, x_18, relu_14, x_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_27.run(arg31_1, buf95, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [x, relu, x_1, relu_1, x_2, x_3, relu_2, x_4, relu_3, x_5, x_6, relu_4, x_7, relu_5, x_8, relu_6, x_9, relu_7, x_10, x_11, relu_8, x_12, relu_9, x_13, relu_10, x_14, relu_11, x_15, x_16, relu_12, x_17, relu_13, x_18, relu_14, x_19], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf96 = extern_kernels.convolution(buf94, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 512, 4, 4), (8192, 1, 2048, 512))
        del buf94
        del buf95
        buf97 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [sub_15, pow_16, mean_15], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_29.run(buf96, arg32_1, buf97, 128, 128, grid=grid(128), stream=stream0)
        del arg32_1
        del buf96
        buf98 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub_15, pow_16, mean_15], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_pow_sub_30.run(buf97, buf98, 2, 64, grid=grid(2), stream=stream0)
        del buf97
        buf100 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sub, pow_1, loss, sub_1, pow_2, mean_1, loss_1, sub_2, pow_3, mean_2, loss_2, sub_3, pow_4, mean_3, loss_3, sub_4, pow_5, mean_4, loss_4, sub_5, pow_6, mean_5, loss_5, sub_6, pow_7, mean_6, loss_6, sub_7, pow_8, mean_7, loss_7, sub_8, pow_9, mean_8, loss_8, sub_9, pow_10, mean_9, loss_9, sub_10, pow_11, mean_10, loss_10, sub_11, pow_12, mean_11, loss_11, sub_12, pow_13, mean_12, loss_12, sub_13, pow_14, mean_13, loss_13, sub_14, pow_15, mean_14, loss_14, sub_15, pow_16, mean_15, loss_15, truediv], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_pow_sub_32.run(buf100, buf80, buf86, buf92, buf98, buf45, buf48, buf51, buf54, buf57, buf60, buf63, buf66, buf69, buf72, buf75, 1, 2, grid=grid(1), stream=stream0)
        del buf45
        del buf48
        del buf51
        del buf54
        del buf57
        del buf60
        del buf63
        del buf66
        del buf69
        del buf72
        del buf75
        del buf80
        del buf86
        del buf92
        del buf98
    return (buf100, )


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
    arg27_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
