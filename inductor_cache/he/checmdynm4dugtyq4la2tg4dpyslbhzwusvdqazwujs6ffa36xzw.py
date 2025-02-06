# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/c4/cc4imz3vggpxflm4k2ligdp7sr7uxvronipdhyb2lyebvnmf6zdy.py
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
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)
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


# kernel path: inductor_cache/3y/c3ybaagrpyprvu4qucwd5owykjor76dxa5rtm4m76njylkiiywna.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xz/cxzcrugq4kcwxq7htwrwvbkq7zxo3viqlah4bvlf6slq5i3eb3jh.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/hz/chzx3h77zrqvorydcxbpch3enncreicer33qj6fv3q7rdqtfr6rl.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/56/c566avupeidxteeuxk6ld7lcqkuua2o3fp34egglcm55m7cyh6po.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wa/cwakcyfqljnfwggi5ow6vvo2uxjsdagoosjiyizd2u654leqrs3t.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xp/cxpnywofdvdzkikmno3rnnj3kgwsmoepeuahwcdobozo3hvlwkh4.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_10 = async_compile.triton('triton_poi_fused_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4u/c4u6vhawyktr7bp5z2vtm4vzmtuqksios7l7tn4rhovunboffll7.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clfmhugteuysrul7lazpx6ytwi2vhhjw2eubv7ypifdltxa7hjzc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_12 = async_compile.triton('triton_poi_fused_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2*x2 + 8*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpffhjwsgz2me5px5taovxcj4h7r75odgjzzstj3vrlbw5yj5bz5.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6zqowfqrnkxukpmc35h4tdpoy366d6uroa3zbbajpdyksqpeish.py
# Topologically Sorted Source Nodes: [init_maxpool], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   init_maxpool => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_14 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_14(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 16)
    x1 = ((xindex // 64) % 16)
    x0 = (xindex % 64)
    x5 = xindex // 1024
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2112) + x0 + 128*x1 + 4096*x5), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2048) + x0 + 128*x1 + 4096*x5), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-1984) + x0 + 128*x1 + 4096*x5), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 4096*x5), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 4096*x5), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 4096*x5), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (1984 + x0 + 128*x1 + 4096*x5), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2048 + x0 + 128*x1 + 4096*x5), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (2112 + x0 + 128*x1 + 4096*x5), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x6), tmp51, None)
    tl.store(out_ptr1 + (x6), tmp76, None)
''', device_str='cuda')


# kernel path: inductor_cache/fh/cfhd452uirw2c4mhcjp6p5q6gfzbefm2umqpwjf6qvilnjzzxhzk.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ci/cci4at5rhjol22hagwzn4ofkbshoxyk4scci5zfwzlns4ftqf2gz.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11, input_12, add], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add => add_8
#   input_10 => convolution_3
#   input_11 => add_7, mul_10, mul_11, sub_3
#   input_12 => relu_3
#   input_7 => convolution_2
#   input_8 => add_5, mul_7, mul_8, sub_2
#   input_9 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_20, %primals_21, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_2, %relu_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = triton_helpers.maximum(tmp18, tmp34)
    tmp36 = tmp19 + tmp35
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp36, None)
''', device_str='cuda')


# kernel path: inductor_cache/4t/c4ttobbe3bql3k27poc72mcpjxbl5ua667nholszm53irc4lnjra.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18, e1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   e1 => add_13
#   input_16 => convolution_5
#   input_17 => add_12, mul_16, mul_17, sub_5
#   input_18 => relu_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
#   %add_13 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_5, %relu_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ef/cefhab3rgiamiztlfp6miusfr2v7mij7xbvkonvi3ut37fb5ib6i.py
# Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => convolution_6
#   input_20 => add_15, mul_19, mul_20, sub_6
#   input_21 => relu_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %primals_38, %primals_39, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/x4/cx45ncp7i77gw6vitarg4ebwjlc32ih7zwgjq3qn35jcolaifhoj.py
# Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, add_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_20
#   input_22 => convolution_7
#   input_23 => add_17, mul_22, mul_23, sub_7
#   input_24 => relu_7
#   input_25 => convolution_8
#   input_26 => add_19, mul_25, mul_26, sub_8
#   input_27 => relu_8
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_44, %primals_45, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %primals_50, %primals_51, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_7, %relu_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = triton_helpers.maximum(tmp18, tmp34)
    tmp36 = tmp19 + tmp35
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp36, None)
''', device_str='cuda')


# kernel path: inductor_cache/b4/cb4y4bsk7y57lbdezg3vaiy4rez4dbzfkry3276oe44dng5ydzec.py
# Topologically Sorted Source Nodes: [input_31, input_32, input_33, e2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   e2 => add_25
#   input_31 => convolution_10
#   input_32 => add_24, mul_31, mul_32, sub_10
#   input_33 => relu_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_62, %primals_63, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
#   %add_25 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_10, %relu_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjozwwf6idzt2uio5pxo4jptkywc4vgdapibniknd7pc5cczy2t3.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_34 => convolution_11
#   input_35 => add_27, mul_34, mul_35, sub_11
#   input_36 => relu_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_25, %primals_68, %primals_69, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/7m/c7m5zcpqxoumyzoznwpg4zb5vl3fsgxiwct3jvqfb47uqs4npfvd.py
# Topologically Sorted Source Nodes: [input_37, input_38, input_39, input_40, input_41, input_42, add_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_32
#   input_37 => convolution_12
#   input_38 => add_29, mul_37, mul_38, sub_12
#   input_39 => relu_12
#   input_40 => convolution_13
#   input_41 => add_31, mul_40, mul_41, sub_13
#   input_42 => relu_13
# Graph fragment:
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_74, %primals_75, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_25, %primals_80, %primals_81, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_12, %relu_13), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = triton_helpers.maximum(tmp18, tmp34)
    tmp36 = tmp19 + tmp35
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp36, None)
''', device_str='cuda')


# kernel path: inductor_cache/li/clivnr4rjktci764axjuq67xetrduyzo7vrevgrte3tvfvb3exui.py
# Topologically Sorted Source Nodes: [input_46, input_47, input_48, e3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   e3 => add_37
#   input_46 => convolution_15
#   input_47 => add_36, mul_46, mul_47, sub_15
#   input_48 => relu_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_92, %primals_93, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
#   %add_37 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_15, %relu_12), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/24/c24qtm3g4exh3r3lq6gpljlagys5mk2uj5jdx4qy6zhc6yxfcxni.py
# Topologically Sorted Source Nodes: [input_49, input_50, input_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_49 => convolution_16
#   input_50 => add_39, mul_49, mul_50, sub_16
#   input_51 => relu_16
# Graph fragment:
#   %convolution_16 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_37, %primals_98, %primals_99, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_39,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ig/cigf6px4d6z4jixg52wwh2qbo4ayhvikoug2es47357yagf3t27i.py
# Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56, input_57, add_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_44
#   input_52 => convolution_17
#   input_53 => add_41, mul_52, mul_53, sub_17
#   input_54 => relu_17
#   input_55 => convolution_18
#   input_56 => add_43, mul_55, mul_56, sub_18
#   input_57 => relu_18
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %primals_104, %primals_105, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_17 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_37, %primals_110, %primals_111, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_43,), kwargs = {})
#   %add_44 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_17, %relu_18), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = triton_helpers.maximum(tmp18, tmp34)
    tmp36 = tmp19 + tmp35
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
    tl.store(in_out_ptr1 + (x2), tmp22, xmask)
    tl.store(out_ptr1 + (x2), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fz/cfza6mxqdbfc2hnkf2hk3aumu2ujpw27cjdvipqxwjrvajvdq3zw.py
# Topologically Sorted Source Nodes: [input_61, input_62, input_63, e4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   e4 => add_49
#   input_61 => convolution_20
#   input_62 => add_48, mul_61, mul_62, sub_20
#   input_63 => relu_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %primals_122, %primals_123, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_48,), kwargs = {})
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_20, %relu_17), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3n/c3nfxewa5dow5vjedlwvxixslsk4e74v76j5wkqqkrr73mjriskn.py
# Topologically Sorted Source Nodes: [input_64, input_65, input_66], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_64 => convolution_21
#   input_65 => add_51, mul_64, mul_65, sub_21
#   input_66 => relu_21
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_49, %primals_128, %primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_51,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxn7wbml65hnyerjlr6e7pnepokhfigvnqmt2fyriizqekln3wri.py
# Topologically Sorted Source Nodes: [convt_out, batchnormout, deconv], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batchnormout => add_53, mul_67, mul_68, sub_22
#   convt_out => convolution_22
#   deconv => relu_22
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_134, %primals_135, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3o/c3oeqsm2634z4cmh4zfkynrvrzlb25j2fworw4umtkgessqf6o3y.py
# Topologically Sorted Source Nodes: [input_67, input_68, input_69, d4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   d4 => add_56
#   input_67 => convolution_23
#   input_68 => add_55, mul_70, mul_71, sub_23
#   input_69 => relu_23
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %primals_140, %primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_55,), kwargs = {})
#   %add_56 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_23, %add_37), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/zc/czc7ilfvga6jfxfbu44rlc4r5acxoqvzb5fandsjqbxvwzhuwtqv.py
# Topologically Sorted Source Nodes: [input_70, input_71, input_72], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_70 => convolution_24
#   input_71 => add_58, mul_73, mul_74, sub_24
#   input_72 => relu_24
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_56, %primals_146, %primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o6/co6ppiumbbt6tdrwwkp4tdymhwifkzzu3skp4avsn77ja7bxlxk3.py
# Topologically Sorted Source Nodes: [convt_out_1, batchnormout_1, deconv_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batchnormout_1 => add_60, mul_76, mul_77, sub_25
#   convt_out_1 => convolution_25
#   deconv_1 => relu_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %primals_152, %primals_153, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_60,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/4s/c4s7iafwesn74zv73b5dmlkewnhmgttg6qplpao5jm2dsikrevrh.py
# Topologically Sorted Source Nodes: [input_73, input_74, input_75, d3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   d3 => add_63
#   input_73 => convolution_26
#   input_74 => add_62, mul_79, mul_80, sub_26
#   input_75 => relu_26
# Graph fragment:
#   %convolution_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %primals_158, %primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_62,), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_26, %add_25), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/v5/cv5ja552neanu5tso3z7b7tg7ynelv46rs2webxsofwcc5zgyrhe.py
# Topologically Sorted Source Nodes: [input_76, input_77, input_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_76 => convolution_27
#   input_77 => add_65, mul_82, mul_83, sub_27
#   input_78 => relu_27
# Graph fragment:
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_63, %primals_164, %primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qk/cqkysub7z2q23pbwrzjdvzctfs4ckop6guso45ehlnfdqs2pbjve.py
# Topologically Sorted Source Nodes: [convt_out_2, batchnormout_2, deconv_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batchnormout_2 => add_67, mul_85, mul_86, sub_28
#   convt_out_2 => convolution_28
#   deconv_2 => relu_28
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_27, %primals_170, %primals_171, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_67,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/6e/c6eogqov5fgpfa5zeffdkqvrilbr37dw5owirldzn3bopjyd4rqk.py
# Topologically Sorted Source Nodes: [input_79, input_80, input_81, d2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   d2 => add_70
#   input_79 => convolution_29
#   input_80 => add_69, mul_88, mul_89, sub_29
#   input_81 => relu_29
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_28, %primals_176, %primals_177, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
#   %add_70 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_29, %add_13), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cynz77ao5jfyjjkqge3tnx35agg4fixvv5baij52ge3iq3xdbz3u.py
# Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_82 => convolution_30
#   input_83 => add_72, mul_91, mul_92, sub_30
#   input_84 => relu_30
# Graph fragment:
#   %convolution_30 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_70, %primals_182, %primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_72,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6kpcjczarcmvr2qbx4k5lxmnvpxurftydkn5nrsakx34bfz25kz.py
# Topologically Sorted Source Nodes: [convt_out_3, batchnormout_3, deconv_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batchnormout_3 => add_74, mul_94, mul_95, sub_31
#   convt_out_3 => convolution_31
#   deconv_3 => relu_31
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_30, %primals_188, %primals_189, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_74,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdx2jougzvj5bzbibtuoy2ew7glfpi3nyscni4got2ofhfn777ec.py
# Topologically Sorted Source Nodes: [input_85, input_86, input_87], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_85 => convolution_32
#   input_86 => add_76, mul_97, mul_98, sub_32
#   input_87 => relu_32
# Graph fragment:
#   %convolution_32 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_31, %primals_194, %primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %relu_32 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_76,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/av/cavbueflfcnaj7avnyosfwxyxryq5g7cr24qydpqfz7xsxxgmmmy.py
# Topologically Sorted Source Nodes: [convt_out_4, batchnormout_4, final_deconv1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batchnormout_4 => add_78, mul_100, mul_101, sub_33
#   convt_out_4 => convolution_33
#   final_deconv1 => relu_33
# Graph fragment:
#   %convolution_33 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_32, %primals_200, %primals_201, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_78,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/7k/c7khdmskt2l7dgb5hz7tpnl2a2uc66kwd7u37bmdr2yikjznkwip.py
# Topologically Sorted Source Nodes: [convt_out_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   convt_out_5 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_34, %primals_212, %primals_213, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_40 = async_compile.triton('triton_poi_fused_convolution_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/m2/cm2bzjaawz4kie6xaips5j6hqukrgmbf37w3n7lkja33zonu4o42.py
# Topologically Sorted Source Nodes: [batchnormout_5, final_deconv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   batchnormout_5 => add_82, mul_106, mul_107, sub_35
#   final_deconv2 => relu_35
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_285), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_287), kwargs = {})
#   %relu_35 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_82,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_35, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2*x2 + 8192*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp17, ymask)
    tl.store(out_ptr1 + (y0 + 2*x2 + 8192*y1), tmp19, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_98, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_147, (64, ), (1, ))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, ), (1, ))
    assert_size_stride(primals_158, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, ), (1, ))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (32, ), (1, ))
    assert_size_stride(primals_169, (32, ), (1, ))
    assert_size_stride(primals_170, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (32, ), (1, ))
    assert_size_stride(primals_173, (32, ), (1, ))
    assert_size_stride(primals_174, (32, ), (1, ))
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_183, (16, ), (1, ))
    assert_size_stride(primals_184, (16, ), (1, ))
    assert_size_stride(primals_185, (16, ), (1, ))
    assert_size_stride(primals_186, (16, ), (1, ))
    assert_size_stride(primals_187, (16, ), (1, ))
    assert_size_stride(primals_188, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_189, (16, ), (1, ))
    assert_size_stride(primals_190, (16, ), (1, ))
    assert_size_stride(primals_191, (16, ), (1, ))
    assert_size_stride(primals_192, (16, ), (1, ))
    assert_size_stride(primals_193, (16, ), (1, ))
    assert_size_stride(primals_194, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, ), (1, ))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, 2, 2, 2), (8, 4, 2, 1))
    assert_size_stride(primals_213, (2, ), (1, ))
    assert_size_stride(primals_214, (2, ), (1, ))
    assert_size_stride(primals_215, (2, ), (1, ))
    assert_size_stride(primals_216, (2, ), (1, ))
    assert_size_stride(primals_217, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, buf2, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_14, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_14
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_20, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_20
        buf5 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_26, buf5, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_26
        buf6 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_32, buf6, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_32
        buf7 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_38, buf7, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_38
        buf8 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_44, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_44
        buf9 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_50, buf9, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_50
        buf10 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_56, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_56
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_62, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_62
        buf12 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_68, buf12, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_68
        buf13 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_74, buf13, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_74
        buf14 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_80, buf14, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_80
        buf15 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_86, buf15, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_86
        buf16 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_92, buf16, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_92
        buf17 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_98, buf17, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_98
        buf18 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_104, buf18, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_104
        buf19 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_110, buf19, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_110
        buf20 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_116, buf20, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_116
        buf21 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_122, buf21, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_122
        buf22 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_134, buf22, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_134
        buf23 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_152, buf23, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_152
        buf24 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_170, buf24, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_170
        buf25 = empty_strided_cuda((16, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_188, buf25, 256, 9, grid=grid(256, 9), stream=stream0)
        del primals_188
        buf26 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_200, buf26, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_200
        buf27 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_206, buf27, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_206
        buf28 = empty_strided_cuda((32, 2, 2, 2), (8, 1, 4, 2), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_212, buf28, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf30, primals_2, primals_4, primals_5, primals_6, primals_7, buf31, 262144, grid=grid(262144), stream=stream0)
        del primals_2
        del primals_7
        buf32 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf33 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.int8)
        # Topologically Sorted Source Nodes: [init_maxpool], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_14.run(buf31, buf32, buf33, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf35 = buf34; del buf34  # reuse
        buf36 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf35, primals_9, primals_10, primals_11, primals_12, primals_13, buf36, 16384, grid=grid(16384), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf32, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        buf41 = buf40; del buf40  # reuse
        buf42 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9, input_10, input_11, input_12, add], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16.run(buf38, buf41, primals_15, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_25, buf39, buf42, 16384, grid=grid(16384), stream=stream0)
        del primals_15
        del primals_21
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf44 = buf43; del buf43  # reuse
        buf45 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf44, primals_27, primals_28, primals_29, primals_30, primals_31, buf45, 16384, grid=grid(16384), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf47 = buf46; del buf46  # reuse
        buf48 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18, e1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17.run(buf47, buf48, primals_33, primals_34, primals_35, primals_36, primals_37, 16384, grid=grid(16384), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf50 = buf49; del buf49  # reuse
        buf51 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18.run(buf50, primals_39, primals_40, primals_41, primals_42, primals_43, buf51, 8192, grid=grid(8192), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 4, 4), (2048, 1, 512, 128))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf48, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        buf56 = buf55; del buf55  # reuse
        buf57 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24, input_25, input_26, input_27, add_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19.run(buf53, buf56, primals_45, primals_46, primals_47, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_55, buf54, buf57, 8192, grid=grid(8192), stream=stream0)
        del primals_45
        del primals_51
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18.run(buf59, primals_57, primals_58, primals_59, primals_60, primals_61, buf60, 8192, grid=grid(8192), stream=stream0)
        del primals_57
        del primals_61
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf62 = buf61; del buf61  # reuse
        buf63 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32, input_33, e2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20.run(buf62, buf63, primals_63, primals_64, primals_65, primals_66, primals_67, 8192, grid=grid(8192), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf65, primals_69, primals_70, primals_71, primals_72, primals_73, buf66, 4096, grid=grid(4096), stream=stream0)
        del primals_69
        del primals_73
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 256, 2, 2), (1024, 1, 512, 256))
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf63, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf68 = buf67; del buf67  # reuse
        buf69 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        buf71 = buf70; del buf70  # reuse
        buf72 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38, input_39, input_40, input_41, input_42, add_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf68, buf71, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, buf69, buf72, 4096, grid=grid(4096), stream=stream0)
        del primals_75
        del primals_81
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, input_45], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf74, primals_87, primals_88, primals_89, primals_90, primals_91, buf75, 4096, grid=grid(4096), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf77 = buf76; del buf76  # reuse
        buf78 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [input_46, input_47, input_48, e3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_23.run(buf77, buf78, primals_93, primals_94, primals_95, primals_96, primals_97, 4096, grid=grid(4096), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, buf17, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 512, 1, 1), (512, 1, 512, 512))
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, input_50, input_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf80, primals_99, primals_100, primals_101, primals_102, primals_103, buf81, 2048, grid=grid(2048), stream=stream0)
        del primals_103
        del primals_99
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 512, 1, 1), (512, 1, 512, 512))
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf78, buf19, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 512, 1, 1), (512, 1, 512, 512))
        buf83 = buf82; del buf82  # reuse
        buf84 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf86 = buf85; del buf85  # reuse
        buf87 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54, input_55, input_56, input_57, add_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf83, buf86, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_115, buf84, buf87, 2048, grid=grid(2048), stream=stream0)
        del primals_105
        del primals_111
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 512, 1, 1), (512, 1, 512, 512))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, input_59, input_60], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf89, primals_117, primals_118, primals_119, primals_120, primals_121, buf90, 2048, grid=grid(2048), stream=stream0)
        del primals_117
        del primals_121
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 512, 1, 1), (512, 1, 512, 512))
        buf92 = buf91; del buf91  # reuse
        buf93 = reinterpret_tensor(buf84, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62, input_63, e4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26.run(buf92, buf93, primals_123, primals_124, primals_125, primals_126, primals_127, 2048, grid=grid(2048), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 128, 1, 1), (128, 1, 128, 128))
        buf95 = buf94; del buf94  # reuse
        buf96 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, input_65, input_66], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf95, primals_129, primals_130, primals_131, primals_132, primals_133, buf96, 512, grid=grid(512), stream=stream0)
        del primals_129
        del primals_133
        # Topologically Sorted Source Nodes: [convt_out], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf97, (4, 128, 2, 2), (512, 1, 256, 128))
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((4, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Topologically Sorted Source Nodes: [convt_out, batchnormout, deconv], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(buf98, primals_135, primals_136, primals_137, primals_138, primals_139, buf99, 2048, grid=grid(2048), stream=stream0)
        del primals_135
        del primals_139
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf101 = buf100; del buf100  # reuse
        buf102 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_67, input_68, input_69, d4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf101, primals_141, primals_142, primals_143, primals_144, primals_145, buf78, buf102, 4096, grid=grid(4096), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 2, 2), (256, 1, 128, 64))
        buf104 = buf103; del buf103  # reuse
        buf105 = empty_strided_cuda((4, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, input_71, input_72], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(buf104, primals_147, primals_148, primals_149, primals_150, primals_151, buf105, 1024, grid=grid(1024), stream=stream0)
        del primals_147
        del primals_151
        # Topologically Sorted Source Nodes: [convt_out_1], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf106, (4, 64, 4, 4), (1024, 1, 256, 64))
        buf107 = buf106; del buf106  # reuse
        buf108 = empty_strided_cuda((4, 64, 4, 4), (1024, 1, 256, 64), torch.float32)
        # Topologically Sorted Source Nodes: [convt_out_1, batchnormout_1, deconv_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(buf107, primals_153, primals_154, primals_155, primals_156, primals_157, buf108, 4096, grid=grid(4096), stream=stream0)
        del primals_153
        del primals_157
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_73, input_74, input_75, d3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32.run(buf110, primals_159, primals_160, primals_161, primals_162, primals_163, buf63, buf111, 8192, grid=grid(8192), stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 4, 4), (512, 1, 128, 32))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((4, 32, 4, 4), (512, 1, 128, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, input_77, input_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf113, primals_165, primals_166, primals_167, primals_168, primals_169, buf114, 2048, grid=grid(2048), stream=stream0)
        del primals_165
        del primals_169
        # Topologically Sorted Source Nodes: [convt_out_2], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf115, (4, 32, 8, 8), (2048, 1, 256, 32))
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided_cuda((4, 32, 8, 8), (2048, 1, 256, 32), torch.float32)
        # Topologically Sorted Source Nodes: [convt_out_2, batchnormout_2, deconv_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf116, primals_171, primals_172, primals_173, primals_174, primals_175, buf117, 8192, grid=grid(8192), stream=stream0)
        del primals_171
        del primals_175
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf119 = buf118; del buf118  # reuse
        buf120 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_79, input_80, input_81, d2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35.run(buf119, primals_177, primals_178, primals_179, primals_180, primals_181, buf48, buf120, 16384, grid=grid(16384), stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 16, 8, 8), (1024, 1, 128, 16))
        buf122 = buf121; del buf121  # reuse
        buf123 = empty_strided_cuda((4, 16, 8, 8), (1024, 1, 128, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf122, primals_183, primals_184, primals_185, primals_186, primals_187, buf123, 4096, grid=grid(4096), stream=stream0)
        del primals_183
        del primals_187
        # Topologically Sorted Source Nodes: [convt_out_3], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf124, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf125 = buf124; del buf124  # reuse
        buf126 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        # Topologically Sorted Source Nodes: [convt_out_3, batchnormout_3, deconv_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(buf125, primals_189, primals_190, primals_191, primals_192, primals_193, buf126, 16384, grid=grid(16384), stream=stream0)
        del primals_189
        del primals_193
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf128 = buf127; del buf127  # reuse
        buf129 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, input_86, input_87], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_38.run(buf128, primals_195, primals_196, primals_197, primals_198, primals_199, buf129, 65536, grid=grid(65536), stream=stream0)
        del primals_195
        del primals_199
        # Topologically Sorted Source Nodes: [convt_out_4], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf130, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf131 = buf130; del buf130  # reuse
        buf132 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [convt_out_4, batchnormout_4, final_deconv1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(buf131, primals_201, primals_202, primals_203, primals_204, primals_205, buf132, 131072, grid=grid(131072), stream=stream0)
        del primals_201
        del primals_205
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf134 = buf133; del buf133  # reuse
        buf135 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, input_89, input_90], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(buf134, primals_207, primals_208, primals_209, primals_210, primals_211, buf135, 131072, grid=grid(131072), stream=stream0)
        del primals_207
        del primals_211
        # Topologically Sorted Source Nodes: [convt_out_5], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, buf28, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 2, 64, 64), (8192, 1, 128, 2))
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [convt_out_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_40.run(buf137, primals_213, 32768, grid=grid(32768), stream=stream0)
        del primals_213
        buf138 = empty_strided_cuda((4, 2, 64, 64), (8192, 4096, 64, 1), torch.float32)
        buf139 = empty_strided_cuda((4, 2, 64, 64), (8192, 1, 128, 2), torch.bool)
        # Topologically Sorted Source Nodes: [batchnormout_5, final_deconv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_41.run(buf137, primals_214, primals_215, primals_216, primals_217, buf138, buf139, 8, 4096, grid=grid(8, 4096), stream=stream0)
        del primals_217
    return (buf138, buf0, buf1, primals_4, primals_5, primals_6, buf2, primals_10, primals_11, primals_12, buf3, primals_16, primals_17, primals_18, primals_19, buf4, primals_22, primals_23, primals_24, primals_25, buf5, primals_28, primals_29, primals_30, buf6, primals_34, primals_35, primals_36, primals_37, buf7, primals_40, primals_41, primals_42, buf8, primals_46, primals_47, primals_48, primals_49, buf9, primals_52, primals_53, primals_54, primals_55, buf10, primals_58, primals_59, primals_60, buf11, primals_64, primals_65, primals_66, primals_67, buf12, primals_70, primals_71, primals_72, buf13, primals_76, primals_77, primals_78, primals_79, buf14, primals_82, primals_83, primals_84, primals_85, buf15, primals_88, primals_89, primals_90, buf16, primals_94, primals_95, primals_96, primals_97, buf17, primals_100, primals_101, primals_102, buf18, primals_106, primals_107, primals_108, primals_109, buf19, primals_112, primals_113, primals_114, primals_115, buf20, primals_118, primals_119, primals_120, buf21, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, buf22, primals_136, primals_137, primals_138, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_149, primals_150, buf23, primals_154, primals_155, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, buf24, primals_172, primals_173, primals_174, primals_176, primals_178, primals_179, primals_180, primals_181, primals_182, primals_184, primals_185, primals_186, buf25, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, buf26, primals_202, primals_203, primals_204, buf27, primals_208, primals_209, primals_210, buf28, primals_214, primals_215, primals_216, buf30, buf31, buf32, buf33, buf35, buf36, buf38, buf41, buf42, buf44, buf45, buf47, buf48, buf50, buf51, buf53, buf56, buf57, buf59, buf60, buf62, buf63, buf65, buf66, buf68, buf71, buf72, buf74, buf75, buf77, buf78, buf80, buf81, buf83, buf86, buf87, buf89, buf90, buf92, buf93, buf95, buf96, buf98, buf99, buf101, buf102, buf104, buf105, buf107, buf108, buf110, buf111, buf113, buf114, buf116, buf117, buf119, buf120, buf122, buf123, buf125, buf126, buf128, buf129, buf131, buf132, buf134, buf135, buf137, buf139, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 2, 2, 2), (8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
