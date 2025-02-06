# AOT ID: ['105_forward']
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


# kernel path: inductor_cache/es/ceskliounr4qjtgyincvttn5ejgkizg3qcqw3ksye66pyixuv4ke.py
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
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/f3/cf3yvvrx2dp4pn5dwcyzm6qhg7y76yekqxcww4ry23bgzk3jew7k.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/az/cazdt4eac53o47y4abuwbxvsaadhafr2pfig3u32lsk5wu2h4nog.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/z5/cz5lcpzem5jshoxsyxcxgxhiha6q7ktzqnt2hpqcbwodal3x5j3j.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/v2/cv2pmewnv75qopjeemipmicxycwgwldxplrujnsdidwoepu6aupv.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gh/cghhujf65f6hqmpdkwffzgq5p7ps2zo6y7ogslendk7v6lklfb35.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => clamp_max, clamp_min
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_5 = async_compile.triton('triton_poi_fused_convolution_hardtanh_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 246016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wi/cwi4h44bhkzjtnz63wpadcwk6upfmayxxpcijpez4dj5vqhsvllm.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_57 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution, 0.0), kwargs = {})
#   %ge_57 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution, 6.0), kwargs = {})
#   %bitwise_or_57 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_57, %ge_57), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_6 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 246016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bv/cbvpvz4mvjniezjfrgztvblslm3ixhn7zeofzen66hdelkv6t6kc.py
# Topologically Sorted Source Nodes: [x, hardtanh_1], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   hardtanh_1 => clamp_max_1, clamp_min_1
#   x => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_4, %primals_5, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_1, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_7 = async_compile.triton('triton_poi_fused_convolution_hardtanh_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4x/c4xqy4z2au546cwdx3npohjoael4cci5eed76v7hx4d2wa35czx3.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   x => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_4, %primals_5, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_56 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_1, 0.0), kwargs = {})
#   %ge_56 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_1, 6.0), kwargs = {})
#   %bitwise_or_56 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_56, %ge_56), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_8 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/67/c67afsg7jaex3emhyaxg2e7ekvkxnavlku7oksuyroxxf2wosnmf.py
# Topologically Sorted Source Nodes: [out, add, input_5, hardtanh_4], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add => add
#   hardtanh_4 => clamp_max_4, clamp_min_4
#   input_5 => clamp_max_3, clamp_min_3
#   out => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %clamp_max_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%clamp_max_3, 0.0), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_9 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = triton_helpers.maximum(tmp8, tmp5)
    tmp10 = triton_helpers.minimum(tmp9, tmp7)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gc/cgckrr5enhcbuo2ladmwhxijyrl6kwlaem7xlbarwkm6y55gvw4l.py
# Topologically Sorted Source Nodes: [out, add, input_5], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add => add
#   input_5 => clamp_max_3, clamp_min_3
#   out => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %clamp_max_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
#   %le_53 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%clamp_max_3, 0.0), kwargs = {})
#   %ge_53 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%clamp_max_3, 6.0), kwargs = {})
#   %bitwise_or_53 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_53, %ge_53), kwargs = {})
#   %le_54 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add, 0.0), kwargs = {})
#   %ge_54 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add, 6.0), kwargs = {})
#   %bitwise_or_54 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_54, %ge_54), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 <= tmp5
    tmp10 = tmp8 >= tmp7
    tmp11 = tmp9 | tmp10
    tmp12 = tmp4 <= tmp5
    tmp13 = tmp4 >= tmp7
    tmp14 = tmp12 | tmp13
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ho/cho7v5y4cyidfes734zzqawrlm3zsmgvdlqi5ki4cqdxyl3ujesn.py
# Topologically Sorted Source Nodes: [out_3, add_3, branch2], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_3 => add_3
#   branch2 => clamp_max_12, clamp_min_12
#   out_3 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_11, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %clamp_max_10), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_11 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lt/cltihupzusbzkstx4vummzlw437bgogw4wiixlz2y24hiru3e3c5.py
# Topologically Sorted Source Nodes: [out_3, add_3], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_3 => add_3
#   out_3 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_11, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %clamp_max_10), kwargs = {})
#   %le_45 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_3, 0.0), kwargs = {})
#   %ge_45 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_3, 6.0), kwargs = {})
#   %bitwise_or_45 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_45, %ge_45), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_backward_12 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_backward_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_backward_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 6.0
    tmp8 = tmp4 >= tmp7
    tmp9 = tmp6 | tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ll/clllpe4jwu63xcko2ekdiqxsalpemx65mkwxrelgybahzsj3x6kc.py
# Topologically Sorted Source Nodes: [x_1, hardtanh_13], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   hardtanh_13 => clamp_max_13, clamp_min_13
#   x_1 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_12, %primals_22, %primals_23, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_10, 0.0), kwargs = {})
#   %clamp_max_13 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_13, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_13 = async_compile.triton('triton_poi_fused_convolution_hardtanh_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/53/c53swvtl4jzj3xt6sygalio2xrte55kqs2aqkpcfpc4tnk5a5doj.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   x_1 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_12, %primals_22, %primals_23, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_44 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_10, 0.0), kwargs = {})
#   %ge_44 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_10, 6.0), kwargs = {})
#   %bitwise_or_44 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_44, %ge_44), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_14 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmjsgx3nffqovyzqljv6hwdfffnzvfqciycvmw3icdu4h5sdrktx.py
# Topologically Sorted Source Nodes: [out_4, add_4, branch3, hardtanh_16], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_4 => add_4
#   branch3 => clamp_max_15, clamp_min_15
#   hardtanh_16 => clamp_max_16, clamp_min_16
#   out_4 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_14, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %clamp_max_13), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6.0), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%clamp_max_15, 0.0), kwargs = {})
#   %clamp_max_16 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_15 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = triton_helpers.maximum(tmp8, tmp5)
    tmp10 = triton_helpers.minimum(tmp9, tmp7)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/om/comokpxrj2ic5xan3aq3k7do5evuf3ynx4k3h44ewjwsbfiv5npn.py
# Topologically Sorted Source Nodes: [out_4, add_4, branch3], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_4 => add_4
#   branch3 => clamp_max_15, clamp_min_15
#   out_4 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_14, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %clamp_max_13), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6.0), kwargs = {})
#   %le_41 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%clamp_max_15, 0.0), kwargs = {})
#   %ge_41 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%clamp_max_15, 6.0), kwargs = {})
#   %bitwise_or_41 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_41, %ge_41), kwargs = {})
#   %le_42 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_4, 0.0), kwargs = {})
#   %ge_42 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_4, 6.0), kwargs = {})
#   %bitwise_or_42 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_42, %ge_42), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_16 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 <= tmp5
    tmp10 = tmp8 >= tmp7
    tmp11 = tmp9 | tmp10
    tmp12 = tmp4 <= tmp5
    tmp13 = tmp4 >= tmp7
    tmp14 = tmp12 | tmp13
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv35pzdxgh3rll7mzs2wj2lj5t4jvtoxmyopqwvc7mbj2vml7g6b.py
# Topologically Sorted Source Nodes: [out_5, add_5, branch4], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_5 => add_5
#   branch4 => clamp_max_18, clamp_min_18
#   out_5 => convolution_14
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_17, %primals_30, %primals_31, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %clamp_max_16), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_17 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv355ho6w5vbfvqn4ucfdy4u2tubmv7z7tnzifrffid626eimvgh.py
# Topologically Sorted Source Nodes: [out_5, add_5], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_5 => add_5
#   out_5 => convolution_14
# Graph fragment:
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_17, %primals_30, %primals_31, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %clamp_max_16), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_5, 0.0), kwargs = {})
#   %ge_39 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_5, 6.0), kwargs = {})
#   %bitwise_or_39 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_39, %ge_39), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_backward_18 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_backward_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_backward_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_backward_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 6.0
    tmp8 = tmp4 >= tmp7
    tmp9 = tmp6 | tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxj7ammp2neh2d7kbpmijpynwksog3o5nzz3pec3ts5rbavjdnfi.py
# Topologically Sorted Source Nodes: [x_2, hardtanh_19], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   hardtanh_19 => clamp_max_19, clamp_min_19
#   x_2 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %primals_32, %primals_33, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_15, 0.0), kwargs = {})
#   %clamp_max_19 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_19 = async_compile.triton('triton_poi_fused_convolution_hardtanh_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwmamozusxtwfqnarcjne5pn7tvnnbwunteu5f7wt7p7b3wcsuw.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   x_2 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %primals_32, %primals_33, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_15, 0.0), kwargs = {})
#   %ge_38 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_15, 6.0), kwargs = {})
#   %bitwise_or_38 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_38, %ge_38), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_20 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wq/cwqaad6qthhfzkpvxrahzlu3wc5qo36yrgquidn46a5sk7a7olrb.py
# Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_48 => convolution_42
#   input_49 => clamp_max_43, clamp_min_43
# Graph fragment:
#   %convolution_42 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %primals_86, %primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_42, 0.0), kwargs = {})
#   %clamp_max_43 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 6.0), kwargs = {})
#   %le_14 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_42, 0.0), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_42, 6.0), kwargs = {})
#   %bitwise_or_14 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_14, %ge_14), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_21 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_21(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xt/cxtpv5oyyctq3gkll3rkcsahq7pbny34km237mrurbocd6g2pc3a.py
# Topologically Sorted Source Nodes: [out_6, add_6, input_20, hardtanh_22], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_6 => add_6
#   hardtanh_22 => clamp_max_22, clamp_min_22
#   input_20 => clamp_max_21, clamp_min_21
#   out_6 => convolution_17
# Graph fragment:
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_20, %primals_36, %primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %clamp_max_19), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_6, 0.0), kwargs = {})
#   %clamp_max_21 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 6.0), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%clamp_max_21, 0.0), kwargs = {})
#   %clamp_max_22 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_22 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = triton_helpers.maximum(tmp8, tmp5)
    tmp10 = triton_helpers.minimum(tmp9, tmp7)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oc/cocrscum4tdmnyq2dyoj3muv67uuvtt424heli6sfq4nz72cz7v6.py
# Topologically Sorted Source Nodes: [out_6, add_6, input_20], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_6 => add_6
#   input_20 => clamp_max_21, clamp_min_21
#   out_6 => convolution_17
# Graph fragment:
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_20, %primals_36, %primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %clamp_max_19), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_6, 0.0), kwargs = {})
#   %clamp_max_21 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 6.0), kwargs = {})
#   %le_35 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%clamp_max_21, 0.0), kwargs = {})
#   %ge_35 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%clamp_max_21, 6.0), kwargs = {})
#   %bitwise_or_35 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_35, %ge_35), kwargs = {})
#   %le_36 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_6, 0.0), kwargs = {})
#   %ge_36 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 6.0), kwargs = {})
#   %bitwise_or_36 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_36, %ge_36), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_23 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 <= tmp5
    tmp10 = tmp8 >= tmp7
    tmp11 = tmp9 | tmp10
    tmp12 = tmp4 <= tmp5
    tmp13 = tmp4 >= tmp7
    tmp14 = tmp12 | tmp13
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pb/cpbcznt5rl7nodxpzrs2vnwkzbg6yfw4mabvt6nqeo6wave3gofe.py
# Topologically Sorted Source Nodes: [out_7, add_7, input_23], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_7 => add_7
#   input_23 => clamp_max_24, clamp_min_24
#   out_7 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_23, %primals_40, %primals_41, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %clamp_max_22), kwargs = {})
#   %clamp_min_24 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_24 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_24 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qe/cqeenfeq5ts6gdunm4xpbxg2n3fonibkvhlndjfyuykpvjz6qo3i.py
# Topologically Sorted Source Nodes: [out_7, add_7], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_7 => add_7
#   out_7 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_23, %primals_40, %primals_41, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %clamp_max_22), kwargs = {})
#   %le_33 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_7, 0.0), kwargs = {})
#   %ge_33 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_7, 6.0), kwargs = {})
#   %bitwise_or_33 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_33, %ge_33), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_backward_25 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_backward_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_backward_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_backward_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 6.0
    tmp8 = tmp4 >= tmp7
    tmp9 = tmp6 | tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a2/ca2sqlco2tnxbkejli4o4o5rkpecuyiouwcld6nui75l6qxvny2s.py
# Topologically Sorted Source Nodes: [x_3, hardtanh_25], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   hardtanh_25 => clamp_max_25, clamp_min_25
#   x_3 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %primals_42, %primals_43, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_25 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_20, 0.0), kwargs = {})
#   %clamp_max_25 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_25, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_26 = async_compile.triton('triton_poi_fused_convolution_hardtanh_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jf/cjfrwhrvfu7lj5gndzrnp6gwv5cabn6eos2tpymh7l5x6g7dus3s.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   x_3 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %primals_42, %primals_43, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_32 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_20, 0.0), kwargs = {})
#   %ge_32 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_20, 6.0), kwargs = {})
#   %bitwise_or_32 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_32, %ge_32), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_27 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y2/cy26yl5bkapyp6b5nxko77gp4wrthhxmj7sbg53yvwbu6gh22b4e.py
# Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_54 => convolution_47
#   input_55 => clamp_max_46, clamp_min_46
# Graph fragment:
#   %convolution_47 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %primals_96, %primals_97, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_46 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_47, 0.0), kwargs = {})
#   %clamp_max_46 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_46, 6.0), kwargs = {})
#   %le_11 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_47, 0.0), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_47, 6.0), kwargs = {})
#   %bitwise_or_11 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_11, %ge_11), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_28 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccsij7htva4pu6fvtshvw4sd3lhcm2hw6x2kkesgjjt6htbgmwng.py
# Topologically Sorted Source Nodes: [out_8, add_8, branch6, hardtanh_28], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_8 => add_8
#   branch6 => clamp_max_27, clamp_min_27
#   hardtanh_28 => clamp_max_28, clamp_min_28
#   out_8 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_26, %primals_46, %primals_47, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %clamp_max_25), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_8, 0.0), kwargs = {})
#   %clamp_max_27 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 6.0), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%clamp_max_27, 0.0), kwargs = {})
#   %clamp_max_28 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_28, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_29 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = triton_helpers.maximum(tmp8, tmp5)
    tmp10 = triton_helpers.minimum(tmp9, tmp7)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yr/cyrwi2r4h5vxr2t3s22ied2rch2lxeoklihaedgrjvuvxurqs3oc.py
# Topologically Sorted Source Nodes: [out_8, add_8, branch6], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_8 => add_8
#   branch6 => clamp_max_27, clamp_min_27
#   out_8 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_26, %primals_46, %primals_47, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %clamp_max_25), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_8, 0.0), kwargs = {})
#   %clamp_max_27 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 6.0), kwargs = {})
#   %le_29 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%clamp_max_27, 0.0), kwargs = {})
#   %ge_29 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%clamp_max_27, 6.0), kwargs = {})
#   %bitwise_or_29 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_29, %ge_29), kwargs = {})
#   %le_30 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_8, 0.0), kwargs = {})
#   %ge_30 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_8, 6.0), kwargs = {})
#   %bitwise_or_30 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_30, %ge_30), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 <= tmp5
    tmp10 = tmp8 >= tmp7
    tmp11 = tmp9 | tmp10
    tmp12 = tmp4 <= tmp5
    tmp13 = tmp4 >= tmp7
    tmp14 = tmp12 | tmp13
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ye/cye6buz4nvahdz3fyxop3etu35tnoqxworzm5l2bwvl4cnaksa62.py
# Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_60 => convolution_52
#   input_61 => clamp_max_49, clamp_min_49
# Graph fragment:
#   %convolution_52 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_28, %primals_106, %primals_107, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_52, 0.0), kwargs = {})
#   %clamp_max_49 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_49, 6.0), kwargs = {})
#   %le_8 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_52, 0.0), kwargs = {})
#   %ge_8 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_52, 6.0), kwargs = {})
#   %bitwise_or_8 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_8, %ge_8), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_31 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_31(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c7/cc7a2ynvehkuotb3dsfawaahvxc3bkchk3hxpzt76qwt4w2pkrjh.py
# Topologically Sorted Source Nodes: [out_10, add_10, branch8], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
# Source node to ATen node mapping:
#   add_10 => add_10
#   branch8 => clamp_max_33, clamp_min_33
#   out_10 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_32, %primals_54, %primals_55, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %clamp_max_31), kwargs = {})
#   %clamp_min_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_10, 0.0), kwargs = {})
#   %clamp_max_33 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_33, 6.0), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_32 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4o/c4opm37j7hgazqlxg62iyki7so32dfk7oui46j45hmktove74ifq.py
# Topologically Sorted Source Nodes: [out_10, add_10], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   add_10 => add_10
#   out_10 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_32, %primals_54, %primals_55, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %clamp_max_31), kwargs = {})
#   %le_24 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_10, 0.0), kwargs = {})
#   %ge_24 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_10, 6.0), kwargs = {})
#   %bitwise_or_24 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_24, %ge_24), kwargs = {})
triton_poi_fused_add_convolution_hardtanh_backward_33 = async_compile.triton('triton_poi_fused_add_convolution_hardtanh_backward_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_hardtanh_backward_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_backward_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 6.0
    tmp8 = tmp4 >= tmp7
    tmp9 = tmp6 | tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4p/c4p3k75wyywywm6ddcj2lewopvy2nxlrt4kwniavxl2llz3gq2wr.py
# Topologically Sorted Source Nodes: [cls_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cls_8 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1, %view_2, %view_3, %view_4, %view_5, %view_6, %view_7], 1), kwargs = {})
triton_poi_fused_cat_34 = async_compile.triton('triton_poi_fused_cat_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1120)
    x1 = xindex // 1120
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 450, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (450*x1 + (((x0) % 450))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0) % 2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 900, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (450*x1 + ((((-450) + x0) % 450))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((((-450) + x0) % 2)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 998, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (98*x1 + ((((-900) + x0) % 98))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr5 + ((((-900) + x0) % 2)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 1096, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr6 + (98*x1 + ((((-998) + x0) % 98))), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr7 + ((((-998) + x0) % 2)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tmp0 >= tmp29
    tmp38 = tl.full([1], 1114, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = tl.load(in_ptr8 + (18*x1 + ((((-1096) + x0) % 18))), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr9 + ((((-1096) + x0) % 2)), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp40, tmp43, tmp44)
    tmp46 = tmp0 >= tmp38
    tmp47 = tl.full([1], 1116, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr10 + (2*x1 + ((-1114) + x0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr11 + ((-1114) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp49, tmp52, tmp53)
    tmp55 = tmp0 >= tmp47
    tmp56 = tl.full([1], 1118, tl.int64)
    tmp57 = tmp0 < tmp56
    tmp58 = tmp55 & tmp57
    tmp59 = tl.load(in_ptr12 + (2*x1 + ((-1116) + x0)), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr13 + ((-1116) + x0), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tmp0 >= tmp56
    tmp65 = tl.full([1], 1120, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tl.load(in_ptr14 + (2*x1 + ((-1118) + x0)), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.load(in_ptr15 + ((-1118) + x0), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp64, tmp69, tmp70)
    tmp72 = tl.where(tmp58, tmp63, tmp71)
    tmp73 = tl.where(tmp49, tmp54, tmp72)
    tmp74 = tl.where(tmp40, tmp45, tmp73)
    tmp75 = tl.where(tmp31, tmp36, tmp74)
    tmp76 = tl.where(tmp22, tmp27, tmp75)
    tmp77 = tl.where(tmp13, tmp18, tmp76)
    tmp78 = tl.where(tmp4, tmp9, tmp77)
    tl.store(out_ptr0 + (x2), tmp78, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfupmsjaf7v5t2sjyvhklmrumujrtpa2i2lfnzptknblzwnsvas.py
# Topologically Sorted Source Nodes: [loc], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   loc => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_8, %view_9, %view_10, %view_11, %view_12, %view_13, %view_14, %view_15], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2240)
    x1 = xindex // 2240
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 900, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (900*x1 + (((x0) % 900))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0) % 4)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1800, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (900*x1 + ((((-900) + x0) % 900))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((((-900) + x0) % 4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 1996, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (196*x1 + ((((-1800) + x0) % 196))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr5 + ((((-1800) + x0) % 4)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 2192, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr6 + (196*x1 + ((((-1996) + x0) % 196))), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr7 + ((((-1996) + x0) % 4)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tmp0 >= tmp29
    tmp38 = tl.full([1], 2228, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = tl.load(in_ptr8 + (36*x1 + ((((-2192) + x0) % 36))), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr9 + ((((-2192) + x0) % 4)), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp40, tmp43, tmp44)
    tmp46 = tmp0 >= tmp38
    tmp47 = tl.full([1], 2232, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr10 + (4*x1 + ((-2228) + x0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr11 + ((-2228) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp49, tmp52, tmp53)
    tmp55 = tmp0 >= tmp47
    tmp56 = tl.full([1], 2236, tl.int64)
    tmp57 = tmp0 < tmp56
    tmp58 = tmp55 & tmp57
    tmp59 = tl.load(in_ptr12 + (4*x1 + ((-2232) + x0)), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr13 + ((-2232) + x0), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tmp0 >= tmp56
    tmp65 = tl.full([1], 2240, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tl.load(in_ptr14 + (4*x1 + ((-2236) + x0)), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.load(in_ptr15 + ((-2236) + x0), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp64, tmp69, tmp70)
    tmp72 = tl.where(tmp58, tmp63, tmp71)
    tmp73 = tl.where(tmp49, tmp54, tmp72)
    tmp74 = tl.where(tmp40, tmp45, tmp73)
    tmp75 = tl.where(tmp31, tmp36, tmp74)
    tmp76 = tl.where(tmp22, tmp27, tmp75)
    tmp77 = tl.where(tmp13, tmp18, tmp76)
    tmp78 = tl.where(tmp4, tmp9, tmp77)
    tl.store(out_ptr0 + (x2), tmp78, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_61, (2, ), (1, ))
    assert_size_stride(primals_62, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_71, (2, ), (1, ))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_75, (4, ), (1, ))
    assert_size_stride(primals_76, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (2, ), (1, ))
    assert_size_stride(primals_82, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_91, (2, ), (1, ))
    assert_size_stride(primals_92, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (2, ), (1, ))
    assert_size_stride(primals_102, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_105, (4, ), (1, ))
    assert_size_stride(primals_106, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (64, ), (1, ))
    assert_size_stride(primals_108, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_111, (2, ), (1, ))
    assert_size_stride(primals_112, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_115, (4, ), (1, ))
    assert_size_stride(primals_116, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_117, (64, ), (1, ))
    assert_size_stride(primals_118, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_119, (64, ), (1, ))
    assert_size_stride(primals_120, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_121, (2, ), (1, ))
    assert_size_stride(primals_122, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_131, (2, ), (1, ))
    assert_size_stride(primals_132, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_135, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_4, buf2, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_4
        buf3 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_6, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_6
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_8
        buf5 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_10, buf5, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_10
        buf6 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf6, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_12
        buf7 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_14, buf7, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_14
        buf8 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_16, buf8, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_16
        buf9 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_18, buf9, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_18
        buf10 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_20, buf10, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_20
        buf11 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_22, buf11, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_22
        buf12 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_24, buf12, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_24
        buf13 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_26, buf13, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_26
        buf14 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_28, buf14, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_28
        buf15 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_30, buf15, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_30
        buf16 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_32, buf16, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_32
        buf17 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_34, buf17, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_34
        buf18 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_36, buf18, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_36
        buf19 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_38, buf19, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_38
        buf20 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_40, buf20, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_40
        buf21 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_42, buf21, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_42
        buf22 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_44, buf22, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_44
        buf23 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_46, buf23, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_46
        buf24 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_48, buf24, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_48
        buf25 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_50, buf25, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_50
        buf26 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_52, buf26, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_52
        buf27 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_54, buf27, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_54
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 31, 31), (61504, 1, 1984, 64))
        buf29 = empty_strided_cuda((4, 64, 31, 31), (61504, 1, 1984, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_5.run(buf28, primals_2, buf29, 246016, grid=grid(246016), stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf205 = empty_strided_cuda((4, 64, 31, 31), (61504, 1, 1984, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_6.run(buf28, primals_2, buf205, 246016, grid=grid(246016), stream=stream0)
        del buf28
        del primals_2
        buf31 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, hardtanh_1], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf30, primals_5, buf31, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf204 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf30, primals_5, buf204, 57600, grid=grid(57600), stream=stream0)
        del primals_5
        buf33 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf32, primals_7, buf33, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf35 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out, add, input_5, hardtanh_4], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_9.run(buf34, primals_9, buf31, buf35, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf201 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        buf202 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out, add, input_5], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10.run(buf34, primals_9, buf31, buf201, buf202, 57600, grid=grid(57600), stream=stream0)
        del primals_9
        buf203 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf32, primals_7, buf203, 57600, grid=grid(57600), stream=stream0)
        del primals_7
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf36, primals_11, buf37, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf39 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_1, add_1, input_8, hardtanh_7], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_9.run(buf38, primals_13, buf35, buf39, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf198 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        buf199 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_1, add_1, input_8], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10.run(buf38, primals_13, buf35, buf198, buf199, 57600, grid=grid(57600), stream=stream0)
        del primals_13
        buf200 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf36, primals_11, buf200, 57600, grid=grid(57600), stream=stream0)
        del primals_11
        buf41 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf40, primals_15, buf41, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf43 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [out_2, add_2, input_11, hardtanh_10], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_9.run(buf42, primals_17, buf39, buf43, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf195 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        buf196 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_2, add_2, input_11], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_10.run(buf42, primals_17, buf39, buf195, buf196, 57600, grid=grid(57600), stream=stream0)
        del primals_17
        buf197 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf40, primals_15, buf197, 57600, grid=grid(57600), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf43, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf45 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf44, primals_19, buf45, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf47 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [out_3, add_3, branch2], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_11.run(buf46, primals_21, buf43, buf47, 57600, grid=grid(57600), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf11, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf193 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_3, add_3], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_backward_12.run(buf46, primals_21, buf43, buf193, 57600, grid=grid(57600), stream=stream0)
        del buf46
        del primals_21
        buf194 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf44, primals_19, buf194, 57600, grid=grid(57600), stream=stream0)
        del primals_19
        buf83 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf82, primals_57, buf83, 57600, grid=grid(57600), stream=stream0)
        buf171 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf82, primals_57, buf171, 57600, grid=grid(57600), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf47, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf49 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, hardtanh_13], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_13.run(buf48, primals_23, buf49, 12544, grid=grid(12544), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf192 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf48, primals_23, buf192, 12544, grid=grid(12544), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 15, 15), (14400, 1, 960, 64))
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf83, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf91 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf90, primals_67, buf91, 57600, grid=grid(57600), stream=stream0)
        buf168 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf90, primals_67, buf168, 57600, grid=grid(57600), stream=stream0)
        del primals_67
        buf51 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_13.run(buf50, primals_25, buf51, 12544, grid=grid(12544), stream=stream0)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf53 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, add_4, branch3, hardtanh_16], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_15.run(buf52, primals_27, buf49, buf53, 12544, grid=grid(12544), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf189 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        buf190 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_4, add_4, branch3], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_16.run(buf52, primals_27, buf49, buf189, buf190, 12544, grid=grid(12544), stream=stream0)
        del primals_27
        buf191 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf50, primals_25, buf191, 12544, grid=grid(12544), stream=stream0)
        del primals_25
        buf85 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf84, primals_59, buf85, 57600, grid=grid(57600), stream=stream0)
        buf170 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf84, primals_59, buf170, 57600, grid=grid(57600), stream=stream0)
        del primals_59
        buf88 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf87, primals_63, buf88, 57600, grid=grid(57600), stream=stream0)
        buf169 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf87, primals_63, buf169, 57600, grid=grid(57600), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 64, 15, 15), (14400, 1, 960, 64))
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf91, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 64, 15, 15), (14400, 1, 960, 64))
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf53, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf55 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_13.run(buf54, primals_29, buf55, 12544, grid=grid(12544), stream=stream0)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf57 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [out_5, add_5, branch4], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_17.run(buf56, primals_31, buf53, buf57, 12544, grid=grid(12544), stream=stream0)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf16, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 128, 3, 3), (1152, 1, 384, 128))
        buf187 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_5, add_5], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_backward_18.run(buf56, primals_31, buf53, buf187, 12544, grid=grid(12544), stream=stream0)
        del buf56
        del primals_31
        buf188 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf54, primals_29, buf188, 12544, grid=grid(12544), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [cls], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 2, 15, 15), (450, 1, 30, 2))
        # Topologically Sorted Source Nodes: [reg], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 4, 15, 15), (900, 1, 60, 4))
        buf93 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf92, primals_69, buf93, 57600, grid=grid(57600), stream=stream0)
        buf167 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf92, primals_69, buf167, 57600, grid=grid(57600), stream=stream0)
        del primals_69
        buf96 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_7.run(buf95, primals_73, buf96, 57600, grid=grid(57600), stream=stream0)
        buf166 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf95, primals_73, buf166, 57600, grid=grid(57600), stream=stream0)
        del buf95
        del primals_73
        buf99 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_13.run(buf98, primals_77, buf99, 12544, grid=grid(12544), stream=stream0)
        buf165 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf98, primals_77, buf165, 12544, grid=grid(12544), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf57, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf59 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, hardtanh_19], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_19.run(buf58, primals_33, buf59, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 128, 3, 3), (1152, 1, 384, 128))
        buf186 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_20.run(buf58, primals_33, buf186, 4608, grid=grid(4608), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [cls_1], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 2, 15, 15), (450, 1, 30, 2))
        # Topologically Sorted Source Nodes: [reg_1], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 4, 15, 15), (900, 1, 60, 4))
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 64, 7, 7), (3136, 1, 448, 64))
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf99, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf107 = buf98; del buf98  # reuse
        buf162 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_21.run(buf106, primals_87, buf107, buf162, 12544, grid=grid(12544), stream=stream0)
        del primals_87
        buf61 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_19.run(buf60, primals_35, buf61, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 3, 3), (1152, 1, 384, 128))
        buf63 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_6, add_6, input_20, hardtanh_22], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_22.run(buf62, primals_37, buf59, buf63, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 128, 3, 3), (1152, 1, 384, 128))
        buf183 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        buf184 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_6, add_6, input_20], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_23.run(buf62, primals_37, buf59, buf183, buf184, 4608, grid=grid(4608), stream=stream0)
        del primals_37
        buf185 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_20.run(buf60, primals_35, buf185, 4608, grid=grid(4608), stream=stream0)
        del primals_35
        buf101 = buf106; del buf106  # reuse
        buf164 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_21.run(buf100, primals_79, buf101, buf164, 12544, grid=grid(12544), stream=stream0)
        del primals_79
        buf104 = buf100; del buf100  # reuse
        buf163 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_21.run(buf103, primals_83, buf104, buf163, 12544, grid=grid(12544), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 64, 7, 7), (3136, 1, 448, 64))
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf107, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 64, 7, 7), (3136, 1, 448, 64))
        buf65 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_19.run(buf64, primals_39, buf65, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 128, 3, 3), (1152, 1, 384, 128))
        buf67 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_7, add_7, input_23], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_24.run(buf66, primals_41, buf63, buf67, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf21, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 1, 1), (128, 1, 128, 128))
        buf181 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_7, add_7], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_backward_25.run(buf66, primals_41, buf63, buf181, 4608, grid=grid(4608), stream=stream0)
        del buf66
        del primals_41
        buf182 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bool)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_20.run(buf64, primals_39, buf182, 4608, grid=grid(4608), stream=stream0)
        del buf64
        del primals_39
        # Topologically Sorted Source Nodes: [cls_2], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 2, 7, 7), (98, 1, 14, 2))
        # Topologically Sorted Source Nodes: [reg_2], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 4, 7, 7), (196, 1, 28, 4))
        buf109 = buf103; del buf103  # reuse
        buf161 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_21.run(buf108, primals_89, buf109, buf161, 12544, grid=grid(12544), stream=stream0)
        del primals_89
        buf112 = buf108; del buf108  # reuse
        buf160 = empty_strided_cuda((4, 64, 7, 7), (3136, 1, 448, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_21.run(buf111, primals_93, buf112, buf160, 12544, grid=grid(12544), stream=stream0)
        del buf111
        del primals_93
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf67, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 64, 3, 3), (576, 1, 192, 64))
        buf69 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, hardtanh_25], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_26.run(buf68, primals_43, buf69, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 128, 1, 1), (128, 1, 128, 128))
        buf180 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_27.run(buf68, primals_43, buf180, 512, grid=grid(512), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [cls_3], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 2, 7, 7), (98, 1, 14, 2))
        # Topologically Sorted Source Nodes: [reg_3], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 4, 7, 7), (196, 1, 28, 4))
        buf115 = empty_strided_cuda((4, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        buf159 = empty_strided_cuda((4, 64, 3, 3), (576, 1, 192, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_28.run(buf114, primals_97, buf115, buf159, 2304, grid=grid(2304), stream=stream0)
        del primals_97
        buf71 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_26.run(buf70, primals_45, buf71, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 128, 1, 1), (128, 1, 128, 128))
        buf73 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, add_8, branch6, hardtanh_28], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_29.run(buf72, primals_47, buf69, buf73, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 1, 1), (128, 1, 128, 128))
        buf177 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        buf178 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_8, add_8, branch6], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30.run(buf72, primals_47, buf69, buf177, buf178, 512, grid=grid(512), stream=stream0)
        del primals_47
        buf179 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_27.run(buf70, primals_45, buf179, 512, grid=grid(512), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 64, 3, 3), (576, 1, 192, 64))
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf115, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 64, 3, 3), (576, 1, 192, 64))
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf73, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 64, 1, 1), (64, 1, 64, 64))
        buf75 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_26.run(buf74, primals_49, buf75, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 128, 1, 1), (128, 1, 128, 128))
        buf77 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [out_9, add_9, branch7, hardtanh_31], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_29.run(buf76, primals_51, buf73, buf77, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 128, 1, 1), (128, 1, 128, 128))
        buf174 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        buf175 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_9, add_9, branch7], Original ATen: [aten.convolution, aten.add, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_30.run(buf76, primals_51, buf73, buf174, buf175, 512, grid=grid(512), stream=stream0)
        del primals_51
        buf176 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_27.run(buf74, primals_49, buf176, 512, grid=grid(512), stream=stream0)
        del primals_49
        buf117 = buf114; del buf114  # reuse
        buf158 = empty_strided_cuda((4, 64, 3, 3), (576, 1, 192, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_28.run(buf116, primals_99, buf117, buf158, 2304, grid=grid(2304), stream=stream0)
        del primals_99
        buf120 = buf116; del buf116  # reuse
        buf157 = empty_strided_cuda((4, 64, 3, 3), (576, 1, 192, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_28.run(buf119, primals_103, buf120, buf157, 2304, grid=grid(2304), stream=stream0)
        del buf119
        del primals_103
        buf123 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf156 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf122, primals_107, buf123, buf156, 256, grid=grid(256), stream=stream0)
        del primals_107
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf77, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 64, 1, 1), (64, 1, 64, 64))
        buf79 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_26.run(buf78, primals_53, buf79, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 128, 1, 1), (128, 1, 128, 128))
        buf81 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [out_10, add_10, branch8], Original ATen: [aten.convolution, aten.add, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_32.run(buf80, primals_55, buf77, buf81, 512, grid=grid(512), stream=stream0)
        buf172 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_10, add_10], Original ATen: [aten.convolution, aten.add, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_backward_33.run(buf80, primals_55, buf77, buf172, 512, grid=grid(512), stream=stream0)
        del buf80
        del primals_55
        buf173 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.bool)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_27.run(buf78, primals_53, buf173, 512, grid=grid(512), stream=stream0)
        del buf78
        del primals_53
        # Topologically Sorted Source Nodes: [cls_4], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 2, 3, 3), (18, 1, 6, 2))
        # Topologically Sorted Source Nodes: [reg_4], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 4, 3, 3), (36, 1, 12, 4))
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 64, 1, 1), (64, 1, 64, 64))
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf123, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 64, 1, 1), (64, 1, 64, 64))
        buf131 = buf122; del buf122  # reuse
        buf153 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf130, primals_117, buf131, buf153, 256, grid=grid(256), stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf81, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 64, 1, 1), (64, 1, 64, 64))
        buf125 = buf130; del buf130  # reuse
        buf155 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf124, primals_109, buf125, buf155, 256, grid=grid(256), stream=stream0)
        del primals_109
        buf128 = buf124; del buf124  # reuse
        buf154 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf127, primals_113, buf128, buf154, 256, grid=grid(256), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 64, 1, 1), (64, 1, 64, 64))
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf131, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 64, 1, 1), (64, 1, 64, 64))
        buf139 = buf127; del buf127  # reuse
        buf150 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_72, input_73], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf138, primals_127, buf139, buf150, 256, grid=grid(256), stream=stream0)
        del primals_127
        # Topologically Sorted Source Nodes: [cls_5], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 2, 1, 1), (2, 1, 2, 2))
        # Topologically Sorted Source Nodes: [reg_5], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 4, 1, 1), (4, 1, 4, 4))
        buf133 = buf138; del buf138  # reuse
        buf152 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf132, primals_119, buf133, buf152, 256, grid=grid(256), stream=stream0)
        del primals_119
        buf136 = buf132; del buf132  # reuse
        buf151 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_70, input_71], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf135, primals_123, buf136, buf151, 256, grid=grid(256), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 64, 1, 1), (64, 1, 64, 64))
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf139, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 64, 1, 1), (64, 1, 64, 64))
        # Topologically Sorted Source Nodes: [cls_6], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 2, 1, 1), (2, 1, 2, 2))
        # Topologically Sorted Source Nodes: [reg_6], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 4, 1, 1), (4, 1, 4, 4))
        buf141 = buf135; del buf135  # reuse
        buf149 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf140, primals_129, buf141, buf149, 256, grid=grid(256), stream=stream0)
        del primals_129
        buf144 = buf140; del buf140  # reuse
        buf148 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_31.run(buf143, primals_133, buf144, buf148, 256, grid=grid(256), stream=stream0)
        del buf143
        del primals_133
        # Topologically Sorted Source Nodes: [cls_7], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 2, 1, 1), (2, 1, 2, 2))
        buf146 = empty_strided_cuda((4, 1120), (1120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cls_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_34.run(buf86, primals_61, buf94, primals_71, buf102, primals_81, buf110, primals_91, buf118, primals_101, buf126, primals_111, buf134, primals_121, buf142, primals_131, buf146, 4480, grid=grid(4480), stream=stream0)
        del buf102
        del buf110
        del buf118
        del buf126
        del buf134
        del buf142
        del buf86
        del buf94
        del primals_101
        del primals_111
        del primals_121
        del primals_131
        del primals_61
        del primals_71
        del primals_81
        del primals_91
        # Topologically Sorted Source Nodes: [reg_7], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 4, 1, 1), (4, 1, 4, 4))
        buf147 = empty_strided_cuda((4, 2240), (2240, 1), torch.float32)
        # Topologically Sorted Source Nodes: [loc], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_35.run(buf89, primals_65, buf97, primals_75, buf105, primals_85, buf113, primals_95, buf121, primals_105, buf129, primals_115, buf137, primals_125, buf145, primals_135, buf147, 8960, grid=grid(8960), stream=stream0)
        del buf105
        del buf113
        del buf121
        del buf129
        del buf137
        del buf145
        del buf89
        del buf97
        del primals_105
        del primals_115
        del primals_125
        del primals_135
        del primals_65
        del primals_75
        del primals_85
        del primals_95
    return (buf146, buf147, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf88, buf91, buf93, buf96, buf99, buf101, buf104, buf107, buf109, buf112, buf115, buf117, buf120, buf123, buf125, buf128, buf131, buf133, buf136, buf139, buf141, buf144, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
