# AOT ID: ['12_forward']
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


# kernel path: inductor_cache/ks/cksa7rovu3ro7bt4vrd6ghsjths23wrsv7asy7t26bbd7ihalfqw.py
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
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: inductor_cache/h6/ch6idif7qwul5cpmntwlb6pzuk7uvdh5cx5qbyj7c3k72t6saeli.py
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
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wp/cwpnkmuyaa7brrivkkev22g5hg5cmc7zvm7s3asdfmrlnf7s5bwp.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/az/cazwhu53mdp46y55m4qzth22lih7uv7gs74jcqe2yogbn3owmijr.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ls/clsq4micff7p6ugdzkgdgjtkxcb5plic2csj6selcpiffvrrt4nq.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4c/c4co7el5nf6wufsjj5up6om2ftc5q3xt6bkvsyzw6iatyvnidv2x.py
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
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: inductor_cache/7m/c7myjvnsqunr3fr3zl33trcmkupmn2hj7eo3zdtdu66sb2mdddo4.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/tn/ctnf3g2m64syjrnj4gwwcxfbopmhyfdh7m2747klrbnycsysbng2.py
# Topologically Sorted Source Nodes: [x_1, softplus, tanh, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus => exp, gt, log1p, where
#   tanh => tanh
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => mul_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %tanh), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7h55mztlvdd4ep45ujgtakstj4evvcju47hdbzgppx3njvvu27d.py
# Topologically Sorted Source Nodes: [x_4, softplus_1, tanh_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_1 => exp_1, gt_1, log1p_1, where_1
#   tanh_1 => tanh_1
#   x_4 => add_3, mul_5, mul_6, sub_1
#   x_5 => mul_7
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_3,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 20), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %log1p_1), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_1,), kwargs = {})
#   %mul_7 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %tanh_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw54sxwefx2zifgeezw5buhevov4xfuas4oa7ntquukurnb4sgn6.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_7 => add_5, mul_10, mul_9, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/la/clasuzdbyu7pe7cdmfrlw62hbcf7iwypixb6v3a4b34ldxge7c6w.py
# Topologically Sorted Source Nodes: [x_13, softplus_4, tanh_4, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_4 => exp_4, gt_4, log1p_4, where_4
#   tanh_4 => tanh_4
#   x_13 => add_9, mul_17, mul_18, sub_4
#   x_14 => mul_19
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_9,), kwargs = {})
#   %log1p_4 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_4,), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 20), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_9, %log1p_4), kwargs = {})
#   %tanh_4 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_4,), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %tanh_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4pzb4tx3mkqdqsq2twkobvmv3zrp7dugjewdnj3lbiex7cofmh.py
# Topologically Sorted Source Nodes: [x_16, softplus_5, tanh_5, x_17, x6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   softplus_5 => exp_5, gt_5, log1p_5, where_5
#   tanh_5 => tanh_5
#   x6 => add_12
#   x_16 => add_11, mul_21, mul_22, sub_5
#   x_17 => mul_23
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %exp_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_11,), kwargs = {})
#   %log1p_5 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_5,), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 20), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_11, %log1p_5), kwargs = {})
#   %tanh_5 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_5,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %tanh_5), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %mul_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/ry/cryd7tyjoge7kldpezyrsjt2o5ia2nbls3mvjrsjpirewvkwhrea.py
# Topologically Sorted Source Nodes: [x7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x7 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_27, %mul_11], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 128, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/j2/cj2ygnsqcfq7s7osvfjuimh3kitnyq3oi4lyn2afmoeq5b2d2ebv.py
# Topologically Sorted Source Nodes: [x_25, softplus_8, tanh_8, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_8 => exp_8, gt_8, log1p_8, where_8
#   tanh_8 => tanh_8
#   x_25 => add_18, mul_33, mul_34, sub_8
#   x_26 => mul_35
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_69), kwargs = {})
#   %add_18 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_71), kwargs = {})
#   %exp_8 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_18,), kwargs = {})
#   %log1p_8 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_8,), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_18, 20), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_18, %log1p_8), kwargs = {})
#   %tanh_8 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_8,), kwargs = {})
#   %mul_35 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %tanh_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/jn/cjn6u3lmiexqxtz64rfc4qc7lajuzurlsxyqgytqgwblrlxwm4zi.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_28 => add_20, mul_37, mul_38, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_77), kwargs = {})
#   %add_20 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_79), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2jqtssde4dydpumfrv3t6dixkrmyrm6pzpuvvxmyy3qmcr5g2na.py
# Topologically Sorted Source Nodes: [x_31, softplus_10, tanh_10, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_10 => exp_10, gt_10, log1p_10, where_10
#   tanh_10 => tanh_10
#   x_31 => add_22, mul_41, mul_42, sub_10
#   x_32 => mul_43
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_85), kwargs = {})
#   %add_22 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %unsqueeze_87), kwargs = {})
#   %exp_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_22,), kwargs = {})
#   %log1p_10 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_10,), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_22, 20), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %add_22, %log1p_10), kwargs = {})
#   %tanh_10 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_10,), kwargs = {})
#   %mul_43 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %tanh_10), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/im/cimjk6frt7aictltu2laeazup2aemks2m2fuetwz2ywixgwiia5a.py
# Topologically Sorted Source Nodes: [x_37, softplus_12, tanh_12, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   softplus_12 => exp_12, gt_12, log1p_12, where_12
#   tanh_12 => tanh_12
#   x_37 => add_26, mul_49, mul_50, sub_12
#   x_38 => mul_51
#   x_39 => add_27
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_101), kwargs = {})
#   %add_26 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_103), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_26,), kwargs = {})
#   %log1p_12 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_12,), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_26, 20), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_26, %log1p_12), kwargs = {})
#   %tanh_12 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_12,), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %tanh_12), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %mul_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = 20.0
    tmp18 = tmp15 > tmp17
    tmp19 = tl_math.exp(tmp15)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp15, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/ke/ckee7q5hghhiybpvdiwlrwksjzlgypfrksbdazifb2yvqjdcw2ep.py
# Topologically Sorted Source Nodes: [x4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x4 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_63, %mul_39], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 128, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf74l7s4rhvw6pxqzbytt2zys4hjsv7ew62czm2r6z7defitxofq.py
# Topologically Sorted Source Nodes: [x_54, softplus_17, tanh_17, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_17 => exp_17, gt_17, log1p_17, where_17
#   tanh_17 => tanh_17
#   x_54 => add_38, mul_69, mul_70, sub_17
#   x_55 => mul_71
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %unsqueeze_141), kwargs = {})
#   %add_38 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %unsqueeze_143), kwargs = {})
#   %exp_17 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_38,), kwargs = {})
#   %log1p_17 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_17,), kwargs = {})
#   %gt_17 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_38, 20), kwargs = {})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_17, %add_38, %log1p_17), kwargs = {})
#   %tanh_17 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_17,), kwargs = {})
#   %mul_71 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %tanh_17), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/24/c24ulenygghvqya4672oyvquw5lodzd2pumi7ppdkgx5wxul7ngl.py
# Topologically Sorted Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_57 => add_40, mul_73, mul_74, sub_18
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_149), kwargs = {})
#   %add_40 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_151), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7l/c7lkz52ajajfb7256h23dcms5q5p3kvet6e4k64wtm2l5xwrfeud.py
# Topologically Sorted Source Nodes: [x_60, softplus_19, tanh_19, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_19 => exp_19, gt_19, log1p_19, where_19
#   tanh_19 => tanh_19
#   x_60 => add_42, mul_77, mul_78, sub_19
#   x_61 => mul_79
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %unsqueeze_157), kwargs = {})
#   %add_42 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %unsqueeze_159), kwargs = {})
#   %exp_19 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_42,), kwargs = {})
#   %log1p_19 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_19,), kwargs = {})
#   %gt_19 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_42, 20), kwargs = {})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %add_42, %log1p_19), kwargs = {})
#   %tanh_19 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_19,), kwargs = {})
#   %mul_79 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %tanh_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/cg/ccg2tzzzxjocpxp7yfhdrxcyunp7elln2dady6mirt7lbbqlu4tx.py
# Topologically Sorted Source Nodes: [x_66, softplus_21, tanh_21, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   softplus_21 => exp_21, gt_21, log1p_21, where_21
#   tanh_21 => tanh_21
#   x_66 => add_46, mul_85, mul_86, sub_21
#   x_67 => mul_87
#   x_68 => add_47
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_173), kwargs = {})
#   %add_46 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_175), kwargs = {})
#   %exp_21 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_46,), kwargs = {})
#   %log1p_21 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_21,), kwargs = {})
#   %gt_21 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_46, 20), kwargs = {})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_21, %add_46, %log1p_21), kwargs = {})
#   %tanh_21 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_21,), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, %tanh_21), kwargs = {})
#   %add_47 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %mul_87), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = 20.0
    tmp18 = tmp15 > tmp17
    tmp19 = tl_math.exp(tmp15)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp15, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/5m/c5mwadym4h4nlufamsczwkkwl7r77ykhrtc7d6f7367ypl3l4ezt.py
# Topologically Sorted Source Nodes: [x4_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x4_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_147, %mul_75], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 256, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (128*x1 + ((-128) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/t2/ct2t2ulclie6kxnuu3befgwm6q3zsmyaojzxtwfvmfekstukseyy.py
# Topologically Sorted Source Nodes: [x_125, softplus_38, tanh_38, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_38 => exp_38, gt_38, log1p_38, where_38
#   tanh_38 => tanh_38
#   x_125 => add_88, mul_153, mul_154, sub_38
#   x_126 => mul_155
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %unsqueeze_309), kwargs = {})
#   %add_88 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %unsqueeze_311), kwargs = {})
#   %exp_38 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_88,), kwargs = {})
#   %log1p_38 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_38,), kwargs = {})
#   %gt_38 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_88, 20), kwargs = {})
#   %where_38 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %add_88, %log1p_38), kwargs = {})
#   %tanh_38 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_38,), kwargs = {})
#   %mul_155 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_88, %tanh_38), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/ez/cezypewall6q3zepgznvxxu3gdf4woqqsz7cpot4u2zv2ekdb27i.py
# Topologically Sorted Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_128 => add_90, mul_157, mul_158, sub_39
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_317), kwargs = {})
#   %add_90 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_319), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ds/cdsnbz567upb45vk5q32i3vlpfz7sbrhdj22fydlyuzp2odvrtnb.py
# Topologically Sorted Source Nodes: [x_131, softplus_40, tanh_40, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_40 => exp_40, gt_40, log1p_40, where_40
#   tanh_40 => tanh_40
#   x_131 => add_92, mul_161, mul_162, sub_40
#   x_132 => mul_163
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_321), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_161, %unsqueeze_325), kwargs = {})
#   %add_92 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %unsqueeze_327), kwargs = {})
#   %exp_40 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_92,), kwargs = {})
#   %log1p_40 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_40,), kwargs = {})
#   %gt_40 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_92, 20), kwargs = {})
#   %where_40 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_40, %add_92, %log1p_40), kwargs = {})
#   %tanh_40 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_40,), kwargs = {})
#   %mul_163 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %tanh_40), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwmvpwyookjnczaz2qybro4asqqmd4co2uctmdwfiu5pqwrjw3i.py
# Topologically Sorted Source Nodes: [x_137, softplus_42, tanh_42, x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   softplus_42 => exp_42, gt_42, log1p_42, where_42
#   tanh_42 => tanh_42
#   x_137 => add_96, mul_169, mul_170, sub_42
#   x_138 => mul_171
#   x_139 => add_97
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_341), kwargs = {})
#   %add_96 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_343), kwargs = {})
#   %exp_42 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_96,), kwargs = {})
#   %log1p_42 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_42,), kwargs = {})
#   %gt_42 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_96, 20), kwargs = {})
#   %where_42 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_42, %add_96, %log1p_42), kwargs = {})
#   %tanh_42 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_42,), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, %tanh_42), kwargs = {})
#   %add_97 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %mul_171), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = 20.0
    tmp18 = tmp15 > tmp17
    tmp19 = tl_math.exp(tmp15)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp15, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/bl/cblg5efsrjb4obpojoixnm5bfjzwvsb6q6ebhtfnxrqx5z4tbdki.py
# Topologically Sorted Source Nodes: [x4_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x4_2 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_231, %mul_159], 1), kwargs = {})
triton_poi_fused_cat_31 = async_compile.triton('triton_poi_fused_cat_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 512, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (256*x1 + ((-256) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/ty/ctywlcxial4wlmh7cbkn4jxoie7ledrgspxads67hpc3ziisdyxl.py
# Topologically Sorted Source Nodes: [x_196, softplus_59, tanh_59, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_59 => exp_59, gt_59, log1p_59, where_59
#   tanh_59 => tanh_59
#   x_196 => add_138, mul_237, mul_238, sub_59
#   x_197 => mul_239
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_473), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_477), kwargs = {})
#   %add_138 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_479), kwargs = {})
#   %exp_59 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_138,), kwargs = {})
#   %log1p_59 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_59,), kwargs = {})
#   %gt_59 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_138, 20), kwargs = {})
#   %where_59 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_59, %add_138, %log1p_59), kwargs = {})
#   %tanh_59 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_59,), kwargs = {})
#   %mul_239 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_138, %tanh_59), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/cr/ccrzgm6sewzkfzsrbn25iyh7mifuxqlcdjsg57qjl6pi3fgpgqp5.py
# Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_199 => add_140, mul_241, mul_242, sub_60
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_481), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_485), kwargs = {})
#   %add_140 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_487), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ni/cni3eb6ahsmis6lkw7bpifwixveayx7ijn4gdtbxe26jpqwnf7c3.py
# Topologically Sorted Source Nodes: [x_202, softplus_61, tanh_61, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_61 => exp_61, gt_61, log1p_61, where_61
#   tanh_61 => tanh_61
#   x_202 => add_142, mul_245, mul_246, sub_61
#   x_203 => mul_247
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %unsqueeze_493), kwargs = {})
#   %add_142 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_495), kwargs = {})
#   %exp_61 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_142,), kwargs = {})
#   %log1p_61 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_61,), kwargs = {})
#   %gt_61 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_142, 20), kwargs = {})
#   %where_61 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_61, %add_142, %log1p_61), kwargs = {})
#   %tanh_61 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_61,), kwargs = {})
#   %mul_247 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %tanh_61), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/ul/culiw6qhjtkqzsnbgelikgzztm2ur5x3243pzkczceyov3emd5td.py
# Topologically Sorted Source Nodes: [x_208, softplus_63, tanh_63, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   softplus_63 => exp_63, gt_63, log1p_63, where_63
#   tanh_63 => tanh_63
#   x_208 => add_146, mul_253, mul_254, sub_63
#   x_209 => mul_255
#   x_210 => add_147
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_509), kwargs = {})
#   %add_146 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_511), kwargs = {})
#   %exp_63 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_146,), kwargs = {})
#   %log1p_63 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_63,), kwargs = {})
#   %gt_63 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_146, 20), kwargs = {})
#   %where_63 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_63, %add_146, %log1p_63), kwargs = {})
#   %tanh_63 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_63,), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_146, %tanh_63), kwargs = {})
#   %add_147 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %mul_255), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = 20.0
    tmp18 = tmp15 > tmp17
    tmp19 = tl_math.exp(tmp15)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp15, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/th/cthj3375tsx76cjkbc27v2ze4fvzvlmwin5tkq7rvshue26hie73.py
# Topologically Sorted Source Nodes: [x4_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x4_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_283, %mul_243], 1), kwargs = {})
triton_poi_fused_cat_36 = async_compile.triton('triton_poi_fused_cat_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 1024, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (512*x1 + ((-512) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/3n/c3nplfbcaqipbiltegca4das3m3h5z4wddbcci5kvjggct5oxfpp.py
# Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_239 => add_168, mul_289, mul_290, sub_72
#   x_240 => gt_72, mul_291, where_72
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_581), kwargs = {})
#   %add_168 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_583), kwargs = {})
#   %gt_72 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_168, 0), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, 0.1), kwargs = {})
#   %where_72 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_72, %add_168, %mul_291), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbjqg3la7bj3y4duip3a4ky746cohziykajbyolezcbxn5z36jo.py
# Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_242 => add_170, mul_293, mul_294, sub_73
#   x_243 => gt_73, mul_295, where_73
# Graph fragment:
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_585), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %unsqueeze_589), kwargs = {})
#   %add_170 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_294, %unsqueeze_591), kwargs = {})
#   %gt_73 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_170, 0), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_170, 0.1), kwargs = {})
#   %where_73 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_73, %add_170, %mul_295), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckadnttzh6f7mwrdiqanaujqjs6vxwrftjki4zmpapvbzsgsg5p4.py
# Topologically Sorted Source Nodes: [m1, spp], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
# Source node to ATen node mapping:
#   m1 => _low_memory_max_pool2d_with_offsets, getitem_1
#   spp => cat_5
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%where_74, [5, 5], [1, 1], [2, 2], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_4, %getitem_2, %getitem, %where_74], 1), kwargs = {})
triton_poi_fused_cat_max_pool2d_with_indices_39 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_39(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 2)
    x1 = ((xindex // 512) % 2)
    x7 = xindex
    x0 = (xindex % 512)
    x4 = xindex // 512
    tmp189 = tl.load(in_ptr0 + (x7), None)
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-3072) + x7), tmp10, other=float("-inf"))
    tmp12 = (-1) + x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2560) + x7), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-2048) + x7), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-1536) + x7), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = 2 + x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-1024) + x7), tmp37, other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = (-1) + x2
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-2048) + x7), tmp44, other=float("-inf"))
    tmp46 = triton_helpers.maximum(tmp45, tmp39)
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-1536) + x7), tmp47, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp46)
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + ((-1024) + x7), tmp50, other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp49)
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + ((-512) + x7), tmp53, other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp52)
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + (x7), tmp56, other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = x2
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + ((-1024) + x7), tmp63, other=float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp58)
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + ((-512) + x7), tmp66, other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp65)
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (x7), tmp69, other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp68)
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (512 + x7), tmp72, other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (1024 + x7), tmp75, other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = 1 + x2
    tmp79 = tmp78 >= tmp1
    tmp80 = tmp78 < tmp3
    tmp81 = tmp79 & tmp80
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (x7), tmp82, other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp77)
    tmp85 = tmp81 & tmp15
    tmp86 = tl.load(in_ptr0 + (512 + x7), tmp85, other=float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp84)
    tmp88 = tmp81 & tmp22
    tmp89 = tl.load(in_ptr0 + (1024 + x7), tmp88, other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp87)
    tmp91 = tmp81 & tmp29
    tmp92 = tl.load(in_ptr0 + (1536 + x7), tmp91, other=float("-inf"))
    tmp93 = triton_helpers.maximum(tmp92, tmp90)
    tmp94 = tmp81 & tmp36
    tmp95 = tl.load(in_ptr0 + (2048 + x7), tmp94, other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp93)
    tmp97 = 2 + x2
    tmp98 = tmp97 >= tmp1
    tmp99 = tmp97 < tmp3
    tmp100 = tmp98 & tmp99
    tmp101 = tmp100 & tmp9
    tmp102 = tl.load(in_ptr0 + (1024 + x7), tmp101, other=float("-inf"))
    tmp103 = triton_helpers.maximum(tmp102, tmp96)
    tmp104 = tmp100 & tmp15
    tmp105 = tl.load(in_ptr0 + (1536 + x7), tmp104, other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp103)
    tmp107 = tmp100 & tmp22
    tmp108 = tl.load(in_ptr0 + (2048 + x7), tmp107, other=float("-inf"))
    tmp109 = triton_helpers.maximum(tmp108, tmp106)
    tmp110 = tmp100 & tmp29
    tmp111 = tl.load(in_ptr0 + (2560 + x7), tmp110, other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp109)
    tmp113 = tmp100 & tmp36
    tmp114 = tl.load(in_ptr0 + (3072 + x7), tmp113, other=float("-inf"))
    tmp115 = triton_helpers.maximum(tmp114, tmp112)
    tmp116 = tmp17 > tmp11
    tmp117 = tl.full([1], 1, tl.int8)
    tmp118 = tl.full([1], 0, tl.int8)
    tmp119 = tl.where(tmp116, tmp117, tmp118)
    tmp120 = tmp24 > tmp18
    tmp121 = tl.full([1], 2, tl.int8)
    tmp122 = tl.where(tmp120, tmp121, tmp119)
    tmp123 = tmp31 > tmp25
    tmp124 = tl.full([1], 3, tl.int8)
    tmp125 = tl.where(tmp123, tmp124, tmp122)
    tmp126 = tmp38 > tmp32
    tmp127 = tl.full([1], 4, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp125)
    tmp129 = tmp45 > tmp39
    tmp130 = tl.full([1], 5, tl.int8)
    tmp131 = tl.where(tmp129, tmp130, tmp128)
    tmp132 = tmp48 > tmp46
    tmp133 = tl.full([1], 6, tl.int8)
    tmp134 = tl.where(tmp132, tmp133, tmp131)
    tmp135 = tmp51 > tmp49
    tmp136 = tl.full([1], 7, tl.int8)
    tmp137 = tl.where(tmp135, tmp136, tmp134)
    tmp138 = tmp54 > tmp52
    tmp139 = tl.full([1], 8, tl.int8)
    tmp140 = tl.where(tmp138, tmp139, tmp137)
    tmp141 = tmp57 > tmp55
    tmp142 = tl.full([1], 9, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp140)
    tmp144 = tmp64 > tmp58
    tmp145 = tl.full([1], 10, tl.int8)
    tmp146 = tl.where(tmp144, tmp145, tmp143)
    tmp147 = tmp67 > tmp65
    tmp148 = tl.full([1], 11, tl.int8)
    tmp149 = tl.where(tmp147, tmp148, tmp146)
    tmp150 = tmp70 > tmp68
    tmp151 = tl.full([1], 12, tl.int8)
    tmp152 = tl.where(tmp150, tmp151, tmp149)
    tmp153 = tmp73 > tmp71
    tmp154 = tl.full([1], 13, tl.int8)
    tmp155 = tl.where(tmp153, tmp154, tmp152)
    tmp156 = tmp76 > tmp74
    tmp157 = tl.full([1], 14, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp155)
    tmp159 = tmp83 > tmp77
    tmp160 = tl.full([1], 15, tl.int8)
    tmp161 = tl.where(tmp159, tmp160, tmp158)
    tmp162 = tmp86 > tmp84
    tmp163 = tl.full([1], 16, tl.int8)
    tmp164 = tl.where(tmp162, tmp163, tmp161)
    tmp165 = tmp89 > tmp87
    tmp166 = tl.full([1], 17, tl.int8)
    tmp167 = tl.where(tmp165, tmp166, tmp164)
    tmp168 = tmp92 > tmp90
    tmp169 = tl.full([1], 18, tl.int8)
    tmp170 = tl.where(tmp168, tmp169, tmp167)
    tmp171 = tmp95 > tmp93
    tmp172 = tl.full([1], 19, tl.int8)
    tmp173 = tl.where(tmp171, tmp172, tmp170)
    tmp174 = tmp102 > tmp96
    tmp175 = tl.full([1], 20, tl.int8)
    tmp176 = tl.where(tmp174, tmp175, tmp173)
    tmp177 = tmp105 > tmp103
    tmp178 = tl.full([1], 21, tl.int8)
    tmp179 = tl.where(tmp177, tmp178, tmp176)
    tmp180 = tmp108 > tmp106
    tmp181 = tl.full([1], 22, tl.int8)
    tmp182 = tl.where(tmp180, tmp181, tmp179)
    tmp183 = tmp111 > tmp109
    tmp184 = tl.full([1], 23, tl.int8)
    tmp185 = tl.where(tmp183, tmp184, tmp182)
    tmp186 = tmp114 > tmp112
    tmp187 = tl.full([1], 24, tl.int8)
    tmp188 = tl.where(tmp186, tmp187, tmp185)
    tl.store(out_ptr0 + (x0 + 2048*x4), tmp115, None)
    tl.store(out_ptr1 + (x7), tmp188, None)
    tl.store(out_ptr2 + (x0 + 2048*x4), tmp189, None)
''', device_str='cuda')


# kernel path: inductor_cache/2h/c2hkw7qrcscswzxpcl42byyeqa7rlhojtiqw3k7h2xk7bd27kkuz.py
# Topologically Sorted Source Nodes: [spp], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   spp => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_4, %getitem_2, %getitem, %where_74], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 2048*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2m3rrfzregauyefxwhgigehho2rajeobzr4n4ncuv4ydwbqp4w.py
# Topologically Sorted Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_257 => add_180, mul_313, mul_314, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_629), kwargs = {})
#   %add_180 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/af/cafus3earmt3xqlcbn3ck6lggegjsbfskexqkrfhwk2txwrehsxf.py
# Topologically Sorted Source Nodes: [up], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up => add_181, add_182, convert_element_type_158, convert_element_type_159, iota, mul_316, mul_317
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_316, 0), kwargs = {})
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_181, torch.float32), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_158, 0.0), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, 0.5), kwargs = {})
#   %convert_element_type_159 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_317, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_42 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv3rbtlihm7iw24bxngoxae3itmtf4s7jjajek6d6mai3illobnd.py
# Topologically Sorted Source Nodes: [x8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x8 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_79, %_unsafe_index], 1), kwargs = {})
triton_poi_fused_cat_43 = async_compile.triton('triton_poi_fused_cat_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x4 = xindex // 512
    x2 = ((xindex // 2048) % 4)
    x1 = ((xindex // 512) % 4)
    x3 = xindex // 8192
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.1
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 512, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full([XBLOCK], 2, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tmp21 = tl.load(in_ptr1 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr2 + (256*tmp24 + 512*tmp20 + 1024*x3 + ((-256) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.1
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp13, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp12, tmp32)
    tl.store(out_ptr0 + (x6), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/s4/cs4e6zux3wdbrdzh5vjxj4qxfcyislwplp6dpgnwsbpdui6mxlsr.py
# Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_263 => add_188, mul_325, mul_326, sub_80
#   x_264 => gt_80, mul_327, where_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_642), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_644), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_646), kwargs = {})
#   %add_188 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_648), kwargs = {})
#   %gt_80 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_188, 0), kwargs = {})
#   %mul_327 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, 0.1), kwargs = {})
#   %where_80 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_80, %add_188, %mul_327), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfworcxktgjnnn7o4uataysuj3e6bb6qxvmb375g6qbq4g5g4mh5.py
# Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_266 => add_190, mul_329, mul_330, sub_81
#   x_267 => gt_81, mul_331, where_81
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_650), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_652), kwargs = {})
#   %mul_330 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_329, %unsqueeze_654), kwargs = {})
#   %add_190 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_330, %unsqueeze_656), kwargs = {})
#   %gt_81 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_190, 0), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_190, 0.1), kwargs = {})
#   %where_81 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_81, %add_190, %mul_331), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjrjebvlsvsmncstfsbw5ekpaz7lfpknohno37dvq32uyrqulojj.py
# Topologically Sorted Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_278 => add_198, mul_345, mul_346, sub_85
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_682), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_684), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_345, %unsqueeze_686), kwargs = {})
#   %add_198 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_346, %unsqueeze_688), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/fi/cfipwmmhxal4ohituhdlpmj36jx5n7a7e5txzwzmvgtedeolbhkv.py
# Topologically Sorted Source Nodes: [up_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   up_1 => add_199, add_200, convert_element_type_176, convert_element_type_177, iota_2, mul_348, mul_349
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_348 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_348, 0), kwargs = {})
#   %convert_element_type_176 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_199, torch.float32), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_176, 0.0), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_200, 0.5), kwargs = {})
#   %convert_element_type_177 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_349, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_47 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jd/cjdc4hpy5ao33oqbixu4uwle2t3pjogidcvxi2qwqyz64xfwd7fl.py
# Topologically Sorted Source Nodes: [x15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x15 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_86, %_unsafe_index_1], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x2 = ((xindex // 2048) % 8)
    x1 = ((xindex // 256) % 8)
    x3 = xindex // 16384
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.1
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 256, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full([XBLOCK], 4, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tmp21 = tl.load(in_ptr1 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr2 + (128*tmp24 + 512*tmp20 + 2048*x3 + ((-128) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.1
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp13, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp12, tmp32)
    tl.store(out_ptr0 + (x6), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/ed/ceda6da65kqrt67iqfeo4sfv4ub4g7nnn7c7gttq22guply7dif4.py
# Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_284 => add_206, mul_357, mul_358, sub_87
#   x_285 => gt_87, mul_359, where_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_699), kwargs = {})
#   %mul_357 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_701), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_357, %unsqueeze_703), kwargs = {})
#   %add_206 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_358, %unsqueeze_705), kwargs = {})
#   %gt_87 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_206, 0), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_206, 0.1), kwargs = {})
#   %where_87 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_87, %add_206, %mul_359), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/7m/c7msbfcac42nx3sage6fk43xact6or4hhh7adhxrmdszdnnlmtow.py
# Topologically Sorted Source Nodes: [x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_287 => add_208, mul_361, mul_362, sub_88
#   x_288 => gt_88, mul_363, where_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_707), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_709), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_711), kwargs = {})
#   %add_208 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_713), kwargs = {})
#   %gt_88 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_208, 0), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_208, 0.1), kwargs = {})
#   %where_88 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_88, %add_208, %mul_363), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.1
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd7t2esuyce4dplzufbr3qomv6fms6qckxx5oiqvcksvpyvwrof7.py
# Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_301 => convolution_93
# Graph fragment:
#   %convolution_93 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_92, %primals_467, %primals_468, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_51 = async_compile.triton('triton_poi_fused_convolution_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_51(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1020
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 255)
    y1 = yindex // 255
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 255*x2 + 16320*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 64*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/fa/cfaau54ul7nstsqmn2q7wiaukq7rfp2bzp3l36ozldkgnlpvcexd.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x3 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_93, %where_84], 1), kwargs = {})
triton_poi_fused_cat_52 = async_compile.triton('triton_poi_fused_cat_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_52(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.1
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 512, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (256*x1 + ((-256) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp4, tmp12, tmp16)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/q7/cq7mhlfdc32s4umaqsxoj4tp4xjqoceyolnyrg62k5kokm6g7kmw.py
# Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_323 => convolution_101
# Graph fragment:
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_99, %primals_504, %primals_505, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_53 = async_compile.triton('triton_poi_fused_convolution_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_53(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1020
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 255)
    y1 = yindex // 255
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 255*x2 + 4080*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 16*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwr2f3gcp3wegufmhkjgtih33quqbgbcuk5k2xlj7npmo4gxptiq.py
# Topologically Sorted Source Nodes: [x11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x11 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_100, %where_77], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.1
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 1024, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (512*x1 + ((-512) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp4, tmp12, tmp16)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zv/czv2qzurjhgq2hpag7sbb7ffhkpsrgbljbkhjuj3qlnzbgfpkdk6.py
# Topologically Sorted Source Nodes: [x_345], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_345 => convolution_109
# Graph fragment:
#   %convolution_109 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_106, %primals_541, %primals_542, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_55 = async_compile.triton('triton_poi_fused_convolution_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_55(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1020
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 255)
    y1 = yindex // 255
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 255*x2 + 1020*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (256, ), (1, ))
    assert_size_stride(primals_206, (256, ), (1, ))
    assert_size_stride(primals_207, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (256, ), (1, ))
    assert_size_stride(primals_267, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (256, ), (1, ))
    assert_size_stride(primals_270, (256, ), (1, ))
    assert_size_stride(primals_271, (256, ), (1, ))
    assert_size_stride(primals_272, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_278, (256, ), (1, ))
    assert_size_stride(primals_279, (256, ), (1, ))
    assert_size_stride(primals_280, (256, ), (1, ))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_282, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_288, (256, ), (1, ))
    assert_size_stride(primals_289, (256, ), (1, ))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_291, (256, ), (1, ))
    assert_size_stride(primals_292, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_298, (1024, ), (1, ))
    assert_size_stride(primals_299, (1024, ), (1, ))
    assert_size_stride(primals_300, (1024, ), (1, ))
    assert_size_stride(primals_301, (1024, ), (1, ))
    assert_size_stride(primals_302, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (512, ), (1, ))
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (512, ), (1, ))
    assert_size_stride(primals_320, (512, ), (1, ))
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (512, ), (1, ))
    assert_size_stride(primals_325, (512, ), (1, ))
    assert_size_stride(primals_326, (512, ), (1, ))
    assert_size_stride(primals_327, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_328, (512, ), (1, ))
    assert_size_stride(primals_329, (512, ), (1, ))
    assert_size_stride(primals_330, (512, ), (1, ))
    assert_size_stride(primals_331, (512, ), (1, ))
    assert_size_stride(primals_332, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_333, (512, ), (1, ))
    assert_size_stride(primals_334, (512, ), (1, ))
    assert_size_stride(primals_335, (512, ), (1, ))
    assert_size_stride(primals_336, (512, ), (1, ))
    assert_size_stride(primals_337, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (512, ), (1, ))
    assert_size_stride(primals_342, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_343, (512, ), (1, ))
    assert_size_stride(primals_344, (512, ), (1, ))
    assert_size_stride(primals_345, (512, ), (1, ))
    assert_size_stride(primals_346, (512, ), (1, ))
    assert_size_stride(primals_347, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_348, (512, ), (1, ))
    assert_size_stride(primals_349, (512, ), (1, ))
    assert_size_stride(primals_350, (512, ), (1, ))
    assert_size_stride(primals_351, (512, ), (1, ))
    assert_size_stride(primals_352, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (512, ), (1, ))
    assert_size_stride(primals_356, (512, ), (1, ))
    assert_size_stride(primals_357, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_358, (1024, ), (1, ))
    assert_size_stride(primals_359, (1024, ), (1, ))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_363, (512, ), (1, ))
    assert_size_stride(primals_364, (512, ), (1, ))
    assert_size_stride(primals_365, (512, ), (1, ))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_368, (1024, ), (1, ))
    assert_size_stride(primals_369, (1024, ), (1, ))
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_372, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (512, ), (1, ))
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (512, ), (1, ))
    assert_size_stride(primals_377, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_378, (512, ), (1, ))
    assert_size_stride(primals_379, (512, ), (1, ))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (512, ), (1, ))
    assert_size_stride(primals_382, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_383, (1024, ), (1, ))
    assert_size_stride(primals_384, (1024, ), (1, ))
    assert_size_stride(primals_385, (1024, ), (1, ))
    assert_size_stride(primals_386, (1024, ), (1, ))
    assert_size_stride(primals_387, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_393, (256, ), (1, ))
    assert_size_stride(primals_394, (256, ), (1, ))
    assert_size_stride(primals_395, (256, ), (1, ))
    assert_size_stride(primals_396, (256, ), (1, ))
    assert_size_stride(primals_397, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_398, (256, ), (1, ))
    assert_size_stride(primals_399, (256, ), (1, ))
    assert_size_stride(primals_400, (256, ), (1, ))
    assert_size_stride(primals_401, (256, ), (1, ))
    assert_size_stride(primals_402, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_403, (256, ), (1, ))
    assert_size_stride(primals_404, (256, ), (1, ))
    assert_size_stride(primals_405, (256, ), (1, ))
    assert_size_stride(primals_406, (256, ), (1, ))
    assert_size_stride(primals_407, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_408, (512, ), (1, ))
    assert_size_stride(primals_409, (512, ), (1, ))
    assert_size_stride(primals_410, (512, ), (1, ))
    assert_size_stride(primals_411, (512, ), (1, ))
    assert_size_stride(primals_412, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_413, (256, ), (1, ))
    assert_size_stride(primals_414, (256, ), (1, ))
    assert_size_stride(primals_415, (256, ), (1, ))
    assert_size_stride(primals_416, (256, ), (1, ))
    assert_size_stride(primals_417, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_418, (512, ), (1, ))
    assert_size_stride(primals_419, (512, ), (1, ))
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_423, (256, ), (1, ))
    assert_size_stride(primals_424, (256, ), (1, ))
    assert_size_stride(primals_425, (256, ), (1, ))
    assert_size_stride(primals_426, (256, ), (1, ))
    assert_size_stride(primals_427, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_428, (128, ), (1, ))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (128, ), (1, ))
    assert_size_stride(primals_437, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (128, ), (1, ))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_443, (256, ), (1, ))
    assert_size_stride(primals_444, (256, ), (1, ))
    assert_size_stride(primals_445, (256, ), (1, ))
    assert_size_stride(primals_446, (256, ), (1, ))
    assert_size_stride(primals_447, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, ), (1, ))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_453, (256, ), (1, ))
    assert_size_stride(primals_454, (256, ), (1, ))
    assert_size_stride(primals_455, (256, ), (1, ))
    assert_size_stride(primals_456, (256, ), (1, ))
    assert_size_stride(primals_457, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_458, (128, ), (1, ))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_463, (256, ), (1, ))
    assert_size_stride(primals_464, (256, ), (1, ))
    assert_size_stride(primals_465, (256, ), (1, ))
    assert_size_stride(primals_466, (256, ), (1, ))
    assert_size_stride(primals_467, (255, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_468, (255, ), (1, ))
    assert_size_stride(primals_469, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_470, (256, ), (1, ))
    assert_size_stride(primals_471, (256, ), (1, ))
    assert_size_stride(primals_472, (256, ), (1, ))
    assert_size_stride(primals_473, (256, ), (1, ))
    assert_size_stride(primals_474, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_475, (256, ), (1, ))
    assert_size_stride(primals_476, (256, ), (1, ))
    assert_size_stride(primals_477, (256, ), (1, ))
    assert_size_stride(primals_478, (256, ), (1, ))
    assert_size_stride(primals_479, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (512, ), (1, ))
    assert_size_stride(primals_483, (512, ), (1, ))
    assert_size_stride(primals_484, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_485, (256, ), (1, ))
    assert_size_stride(primals_486, (256, ), (1, ))
    assert_size_stride(primals_487, (256, ), (1, ))
    assert_size_stride(primals_488, (256, ), (1, ))
    assert_size_stride(primals_489, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_490, (512, ), (1, ))
    assert_size_stride(primals_491, (512, ), (1, ))
    assert_size_stride(primals_492, (512, ), (1, ))
    assert_size_stride(primals_493, (512, ), (1, ))
    assert_size_stride(primals_494, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_496, (256, ), (1, ))
    assert_size_stride(primals_497, (256, ), (1, ))
    assert_size_stride(primals_498, (256, ), (1, ))
    assert_size_stride(primals_499, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_500, (512, ), (1, ))
    assert_size_stride(primals_501, (512, ), (1, ))
    assert_size_stride(primals_502, (512, ), (1, ))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (255, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_505, (255, ), (1, ))
    assert_size_stride(primals_506, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_507, (512, ), (1, ))
    assert_size_stride(primals_508, (512, ), (1, ))
    assert_size_stride(primals_509, (512, ), (1, ))
    assert_size_stride(primals_510, (512, ), (1, ))
    assert_size_stride(primals_511, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_512, (512, ), (1, ))
    assert_size_stride(primals_513, (512, ), (1, ))
    assert_size_stride(primals_514, (512, ), (1, ))
    assert_size_stride(primals_515, (512, ), (1, ))
    assert_size_stride(primals_516, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_517, (1024, ), (1, ))
    assert_size_stride(primals_518, (1024, ), (1, ))
    assert_size_stride(primals_519, (1024, ), (1, ))
    assert_size_stride(primals_520, (1024, ), (1, ))
    assert_size_stride(primals_521, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_522, (512, ), (1, ))
    assert_size_stride(primals_523, (512, ), (1, ))
    assert_size_stride(primals_524, (512, ), (1, ))
    assert_size_stride(primals_525, (512, ), (1, ))
    assert_size_stride(primals_526, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_527, (1024, ), (1, ))
    assert_size_stride(primals_528, (1024, ), (1, ))
    assert_size_stride(primals_529, (1024, ), (1, ))
    assert_size_stride(primals_530, (1024, ), (1, ))
    assert_size_stride(primals_531, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_532, (512, ), (1, ))
    assert_size_stride(primals_533, (512, ), (1, ))
    assert_size_stride(primals_534, (512, ), (1, ))
    assert_size_stride(primals_535, (512, ), (1, ))
    assert_size_stride(primals_536, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_537, (1024, ), (1, ))
    assert_size_stride(primals_538, (1024, ), (1, ))
    assert_size_stride(primals_539, (1024, ), (1, ))
    assert_size_stride(primals_540, (1024, ), (1, ))
    assert_size_stride(primals_541, (255, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_542, (255, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_27, buf3, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_27
        buf4 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_42, buf4, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_42
        buf5 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_62, buf5, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_62
        buf6 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_72, buf6, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_72
        buf7 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_87, buf7, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_87
        buf8 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_107, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_107
        buf9 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_117, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_117
        buf10 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_127, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_127
        buf11 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_137, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_137
        buf12 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_147, buf12, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_147
        buf13 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_157, buf13, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_157
        buf14 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_167, buf14, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_167
        buf15 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_177, buf15, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_177
        buf16 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_192, buf16, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_192
        buf17 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_212, buf17, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_212
        buf18 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_222, buf18, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_222
        buf19 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_232, buf19, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_232
        buf20 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_242, buf20, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_242
        buf21 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_252, buf21, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_252
        buf22 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_262, buf22, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_262
        buf23 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_272, buf23, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_272
        buf24 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_282, buf24, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_282
        buf25 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_297, buf25, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_297
        buf26 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_317, buf26, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_317
        buf27 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_327, buf27, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_327
        buf28 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_337, buf28, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_337
        buf29 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(primals_347, buf29, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_347
        buf30 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_367, buf30, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_367
        buf31 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_382, buf31, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_382
        buf32 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_407, buf32, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_407
        buf33 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_417, buf33, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_417
        buf34 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_442, buf34, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_442
        buf35 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_452, buf35, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_452
        buf36 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_462, buf36, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_462
        buf37 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_469, buf37, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_469
        buf38 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_479, buf38, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_479
        buf39 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_489, buf39, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_489
        buf40 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_499, buf40, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_499
        buf41 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_506, buf41, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_506
        buf42 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_516, buf42, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_516
        buf43 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_526, buf43, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_526
        buf44 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_536, buf44, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf1, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 32, 64, 64), (131072, 1, 2048, 32))
        buf46 = empty_strided_cuda((4, 32, 64, 64), (131072, 1, 2048, 32), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_1, softplus, tanh, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_11.run(buf47, buf45, primals_3, primals_4, primals_5, primals_6, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf49 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_4, softplus_1, tanh_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12.run(buf50, buf48, primals_8, primals_9, primals_10, primals_11, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf52 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf51, primals_13, primals_14, primals_15, primals_16, buf52, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf50, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf54 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_10, softplus_3, tanh_3, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12.run(buf55, buf53, primals_18, primals_19, primals_20, primals_21, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf57 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_13, softplus_4, tanh_4, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_14.run(buf58, buf56, primals_23, primals_24, primals_25, primals_26, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf60 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_16, softplus_5, tanh_5, x_17, x6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_15.run(buf61, buf59, primals_28, primals_29, primals_30, primals_31, buf55, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf63 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf62, primals_33, primals_34, primals_35, primals_36, buf63, 262144, grid=grid(262144), stream=stream0)
        buf64 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf63, buf52, buf64, 524288, grid=grid(524288), stream=stream0)
        del buf52
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf66 = buf63; del buf63  # reuse
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_22, softplus_7, tanh_7, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_12.run(buf67, buf65, primals_38, primals_39, primals_40, primals_41, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf69 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_25, softplus_8, tanh_8, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17.run(buf70, buf68, primals_43, primals_44, primals_45, primals_46, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf72 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf71, primals_48, primals_49, primals_50, primals_51, buf72, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf70, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf74 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_31, softplus_10, tanh_10, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19.run(buf75, buf73, primals_53, primals_54, primals_55, primals_56, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf77 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_34, softplus_11, tanh_11, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19.run(buf78, buf76, primals_58, primals_59, primals_60, primals_61, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf80 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_37, softplus_12, tanh_12, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20.run(buf81, buf79, primals_63, primals_64, primals_65, primals_66, buf75, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf83 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_41, softplus_13, tanh_13, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_19.run(buf84, buf82, primals_68, primals_69, primals_70, primals_71, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf86 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_44, softplus_14, tanh_14, x_45, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_20.run(buf87, buf85, primals_73, primals_74, primals_75, primals_76, buf81, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf89 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf88, primals_78, primals_79, primals_80, primals_81, buf89, 65536, grid=grid(65536), stream=stream0)
        buf90 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf89, buf72, buf90, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf92 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_51, softplus_16, tanh_16, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_17.run(buf93, buf91, primals_83, primals_84, primals_85, primals_86, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf95 = reinterpret_tensor(buf89, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf89  # reuse
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_54, softplus_17, tanh_17, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22.run(buf96, buf94, primals_88, primals_89, primals_90, primals_91, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf98 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf97, primals_93, primals_94, primals_95, primals_96, buf98, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf96, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf100 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_60, softplus_19, tanh_19, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf101, buf99, primals_98, primals_99, primals_100, primals_101, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf103 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_63, softplus_20, tanh_20, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf104, buf102, primals_103, primals_104, primals_105, primals_106, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf106 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_66, softplus_21, tanh_21, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf107, buf105, primals_108, primals_109, primals_110, primals_111, buf101, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf109 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_70, softplus_22, tanh_22, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf110, buf108, primals_113, primals_114, primals_115, primals_116, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf112 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_73, softplus_23, tanh_23, x_74, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf113, buf111, primals_118, primals_119, primals_120, primals_121, buf107, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf115 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf116 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_77, softplus_24, tanh_24, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf116, buf114, primals_123, primals_124, primals_125, primals_126, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf118 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_80, softplus_25, tanh_25, x_81, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf119, buf117, primals_128, primals_129, primals_130, primals_131, buf113, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf121 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_84, softplus_26, tanh_26, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf122, buf120, primals_133, primals_134, primals_135, primals_136, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf124 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_87, softplus_27, tanh_27, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf125, buf123, primals_138, primals_139, primals_140, primals_141, buf119, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf127 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_91, softplus_28, tanh_28, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf128, buf126, primals_143, primals_144, primals_145, primals_146, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf130 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_94, softplus_29, tanh_29, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf131, buf129, primals_148, primals_149, primals_150, primals_151, buf125, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_97], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf133 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_98, softplus_30, tanh_30, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf134, buf132, primals_153, primals_154, primals_155, primals_156, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf136 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_101, softplus_31, tanh_31, x_102, x_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf137, buf135, primals_158, primals_159, primals_160, primals_161, buf131, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf139 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_105, softplus_32, tanh_32, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf140, buf138, primals_163, primals_164, primals_165, primals_166, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf142 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_108, softplus_33, tanh_33, x_109, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf143, buf141, primals_168, primals_169, primals_170, primals_171, buf137, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf145 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_112, softplus_34, tanh_34, x_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_24.run(buf146, buf144, primals_173, primals_174, primals_175, primals_176, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf148 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_115, softplus_35, tanh_35, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_25.run(buf149, buf147, primals_178, primals_179, primals_180, primals_181, buf143, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf151 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf150, primals_183, primals_184, primals_185, primals_186, buf151, 32768, grid=grid(32768), stream=stream0)
        buf152 = reinterpret_tensor(buf72, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [x4_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf151, buf98, buf152, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf154 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_122, softplus_37, tanh_37, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_22.run(buf155, buf153, primals_188, primals_189, primals_190, primals_191, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, buf16, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf157 = reinterpret_tensor(buf98, (4, 512, 4, 4), (8192, 1, 2048, 512), 0); del buf98  # reuse
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_125, softplus_38, tanh_38, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27.run(buf158, buf156, primals_193, primals_194, primals_195, primals_196, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf160 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf159, primals_198, primals_199, primals_200, primals_201, buf160, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf158, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf162 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_131, softplus_40, tanh_40, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf163, buf161, primals_203, primals_204, primals_205, primals_206, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf165 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_134, softplus_41, tanh_41, x_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf166, buf164, primals_208, primals_209, primals_210, primals_211, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf168 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_137, softplus_42, tanh_42, x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf169, buf167, primals_213, primals_214, primals_215, primals_216, buf163, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf171 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_141, softplus_43, tanh_43, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf172, buf170, primals_218, primals_219, primals_220, primals_221, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf174 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_144, softplus_44, tanh_44, x_145, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf175, buf173, primals_223, primals_224, primals_225, primals_226, buf169, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf177 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf178 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_148, softplus_45, tanh_45, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf178, buf176, primals_228, primals_229, primals_230, primals_231, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf180 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_151, softplus_46, tanh_46, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf181, buf179, primals_233, primals_234, primals_235, primals_236, buf175, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf183 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_155, softplus_47, tanh_47, x_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf184, buf182, primals_238, primals_239, primals_240, primals_241, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf186 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_158, softplus_48, tanh_48, x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf187, buf185, primals_243, primals_244, primals_245, primals_246, buf181, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf189 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_162, softplus_49, tanh_49, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf190, buf188, primals_248, primals_249, primals_250, primals_251, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf192 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_165, softplus_50, tanh_50, x_166, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf193, buf191, primals_253, primals_254, primals_255, primals_256, buf187, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf195 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_169, softplus_51, tanh_51, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf196, buf194, primals_258, primals_259, primals_260, primals_261, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf198 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_172, softplus_52, tanh_52, x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf199, buf197, primals_263, primals_264, primals_265, primals_266, buf193, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf201 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_176, softplus_53, tanh_53, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf202, buf200, primals_268, primals_269, primals_270, primals_271, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf204 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_179, softplus_54, tanh_54, x_180, x_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf205, buf203, primals_273, primals_274, primals_275, primals_276, buf199, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf207 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_183, softplus_55, tanh_55, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_29.run(buf208, buf206, primals_278, primals_279, primals_280, primals_281, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf210 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_186, softplus_56, tanh_56, x_187, x_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_30.run(buf211, buf209, primals_283, primals_284, primals_285, primals_286, buf205, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf213 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf212, primals_288, primals_289, primals_290, primals_291, buf213, 16384, grid=grid(16384), stream=stream0)
        buf214 = reinterpret_tensor(buf151, (4, 512, 4, 4), (8192, 1, 2048, 512), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [x4_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf213, buf160, buf214, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf216 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_193, softplus_58, tanh_58, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_27.run(buf217, buf215, primals_293, primals_294, primals_295, primals_296, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf219 = reinterpret_tensor(buf213, (4, 1024, 2, 2), (4096, 1, 2048, 1024), 0); del buf213  # reuse
        buf220 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_196, softplus_59, tanh_59, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32.run(buf220, buf218, primals_298, primals_299, primals_300, primals_301, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf222 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf221, primals_303, primals_304, primals_305, primals_306, buf222, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf220, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf224 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_202, softplus_61, tanh_61, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34.run(buf225, buf223, primals_308, primals_309, primals_310, primals_311, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf227 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf228 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_205, softplus_62, tanh_62, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34.run(buf228, buf226, primals_313, primals_314, primals_315, primals_316, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf230 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_208, softplus_63, tanh_63, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35.run(buf231, buf229, primals_318, primals_319, primals_320, primals_321, buf225, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf233 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_212, softplus_64, tanh_64, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34.run(buf234, buf232, primals_323, primals_324, primals_325, primals_326, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf236 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_215, softplus_65, tanh_65, x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35.run(buf237, buf235, primals_328, primals_329, primals_330, primals_331, buf231, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf239 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_219, softplus_66, tanh_66, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34.run(buf240, buf238, primals_333, primals_334, primals_335, primals_336, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf242 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf243 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_222, softplus_67, tanh_67, x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35.run(buf243, buf241, primals_338, primals_339, primals_340, primals_341, buf237, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf245 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_226, softplus_68, tanh_68, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_34.run(buf246, buf244, primals_343, primals_344, primals_345, primals_346, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf248 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_229, softplus_69, tanh_69, x_230, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_35.run(buf249, buf247, primals_348, primals_349, primals_350, primals_351, buf243, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf251 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf250, primals_353, primals_354, primals_355, primals_356, buf251, 8192, grid=grid(8192), stream=stream0)
        buf252 = reinterpret_tensor(buf160, (4, 1024, 2, 2), (4096, 1, 2048, 1024), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x4_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_36.run(buf251, buf222, buf252, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf254 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf255 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [x_236, softplus_71, tanh_71, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_32.run(buf255, buf253, primals_358, primals_359, primals_360, primals_361, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf257 = buf251; del buf251  # reuse
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf258, buf256, primals_363, primals_364, primals_365, primals_366, 8192, grid=grid(8192), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf260 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38.run(buf261, buf259, primals_368, primals_369, primals_370, primals_371, 16384, grid=grid(16384), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf263 = buf222; del buf222  # reuse
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf264, buf262, primals_373, primals_374, primals_375, primals_376, 8192, grid=grid(8192), stream=stream0)
        del primals_376
        buf276 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        buf265 = reinterpret_tensor(buf276, (4, 512, 2, 2), (8192, 1, 4096, 2048), 1024)  # alias
        buf266 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.int8)
        buf275 = reinterpret_tensor(buf276, (4, 512, 2, 2), (8192, 1, 4096, 2048), 1536)  # alias
        # Topologically Sorted Source Nodes: [m1, spp], Original ATen: [aten.max_pool2d_with_indices, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_max_pool2d_with_indices_39.run(buf264, buf265, buf266, buf275, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [m2], Original ATen: [aten.max_pool2d_with_indices]
        buf267 = torch.ops.aten.max_pool2d_with_indices.default(buf264, [9, 9], [1, 1], [4, 4])
        buf268 = buf267[0]
        buf269 = buf267[1]
        del buf267
        # Topologically Sorted Source Nodes: [m3], Original ATen: [aten.max_pool2d_with_indices]
        buf270 = torch.ops.aten.max_pool2d_with_indices.default(buf264, [13, 13], [1, 1], [6, 6])
        buf271 = buf270[0]
        buf272 = buf270[1]
        del buf270
        buf273 = reinterpret_tensor(buf276, (4, 512, 2, 2), (8192, 1, 4096, 2048), 0)  # alias
        # Topologically Sorted Source Nodes: [spp], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf271, buf273, 8192, grid=grid(8192), stream=stream0)
        buf274 = reinterpret_tensor(buf276, (4, 512, 2, 2), (8192, 1, 4096, 2048), 512)  # alias
        # Topologically Sorted Source Nodes: [spp], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf268, buf274, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf278 = buf268; del buf268  # reuse
        buf279 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf279, buf277, primals_378, primals_379, primals_380, primals_381, 8192, grid=grid(8192), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf281 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38.run(buf282, buf280, primals_383, primals_384, primals_385, primals_386, 16384, grid=grid(16384), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf284 = buf271; del buf271  # reuse
        buf285 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf285, buf283, primals_388, primals_389, primals_390, primals_391, 8192, grid=grid(8192), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf287 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf286, primals_393, primals_394, primals_395, primals_396, buf287, 4096, grid=grid(4096), stream=stream0)
        buf288 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_42.run(buf288, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf217, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf290 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf289, primals_398, primals_399, primals_400, primals_401, buf290, 16384, grid=grid(16384), stream=stream0)
        buf291 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_43.run(buf290, buf288, buf287, buf291, 32768, grid=grid(32768), stream=stream0)
        del buf287
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf293 = buf290; del buf290  # reuse
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf294, buf292, primals_403, primals_404, primals_405, primals_406, 16384, grid=grid(16384), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf296 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf297, buf295, primals_408, primals_409, primals_410, primals_411, 32768, grid=grid(32768), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_412, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf299 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf300, buf298, primals_413, primals_414, primals_415, primals_416, 16384, grid=grid(16384), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf302 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf303 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf303, buf301, primals_418, primals_419, primals_420, primals_421, 32768, grid=grid(32768), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf305 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf306 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf306, buf304, primals_423, primals_424, primals_425, primals_426, 16384, grid=grid(16384), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf308 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf307, primals_428, primals_429, primals_430, primals_431, buf308, 8192, grid=grid(8192), stream=stream0)
        buf309 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_47.run(buf309, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf155, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf311 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf310, primals_433, primals_434, primals_435, primals_436, buf311, 32768, grid=grid(32768), stream=stream0)
        buf312 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf311, buf309, buf308, buf312, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf314 = buf311; del buf311  # reuse
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49.run(buf315, buf313, primals_438, primals_439, primals_440, primals_441, 32768, grid=grid(32768), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [x_286], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf317 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf318 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50.run(buf318, buf316, primals_443, primals_444, primals_445, primals_446, 65536, grid=grid(65536), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf320 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49.run(buf321, buf319, primals_448, primals_449, primals_450, primals_451, 32768, grid=grid(32768), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf323 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf324 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50.run(buf324, buf322, primals_453, primals_454, primals_455, primals_456, 65536, grid=grid(65536), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf326 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_49.run(buf327, buf325, primals_458, primals_459, primals_460, primals_461, 32768, grid=grid(32768), stream=stream0)
        del primals_461
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf329 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_50.run(buf330, buf328, primals_463, primals_464, primals_465, primals_466, 65536, grid=grid(65536), stream=stream0)
        del primals_466
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 255, 8, 8), (16320, 1, 2040, 255))
        buf332 = empty_strided_cuda((4, 255, 8, 8), (16320, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_51.run(buf331, primals_468, buf332, 1020, 64, grid=grid(1020, 64), stream=stream0)
        del buf331
        del primals_468
        # Topologically Sorted Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf327, buf37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf334 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf333, primals_470, primals_471, primals_472, primals_473, buf334, 16384, grid=grid(16384), stream=stream0)
        buf335 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf334, buf306, buf335, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf337 = buf334; del buf334  # reuse
        buf338 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_306, x_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf338, buf336, primals_475, primals_476, primals_477, primals_478, 16384, grid=grid(16384), stream=stream0)
        del primals_478
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf340 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf341, buf339, primals_480, primals_481, primals_482, primals_483, 32768, grid=grid(32768), stream=stream0)
        del primals_483
        # Topologically Sorted Source Nodes: [x_311], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_484, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf343 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf344, buf342, primals_485, primals_486, primals_487, primals_488, 16384, grid=grid(16384), stream=stream0)
        del primals_488
        # Topologically Sorted Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf346 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf347 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf347, buf345, primals_490, primals_491, primals_492, primals_493, 32768, grid=grid(32768), stream=stream0)
        del primals_493
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_494, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf349 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_44.run(buf350, buf348, primals_495, primals_496, primals_497, primals_498, 16384, grid=grid(16384), stream=stream0)
        del primals_498
        # Topologically Sorted Source Nodes: [x_320], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf352 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf353 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf353, buf351, primals_500, primals_501, primals_502, primals_503, 32768, grid=grid(32768), stream=stream0)
        del primals_503
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_504, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 255, 4, 4), (4080, 1, 1020, 255))
        buf355 = empty_strided_cuda((4, 255, 4, 4), (4080, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_53.run(buf354, primals_505, buf355, 1020, 16, grid=grid(1020, 16), stream=stream0)
        del buf354
        del primals_505
        # Topologically Sorted Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf350, buf41, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf357 = reinterpret_tensor(buf308, (4, 512, 2, 2), (2048, 1, 1024, 512), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_325], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf356, primals_507, primals_508, primals_509, primals_510, buf357, 8192, grid=grid(8192), stream=stream0)
        buf358 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf357, buf285, buf358, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_511, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf360 = buf357; del buf357  # reuse
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf361, buf359, primals_512, primals_513, primals_514, primals_515, 8192, grid=grid(8192), stream=stream0)
        del primals_515
        # Topologically Sorted Source Nodes: [x_330], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf363 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf364 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38.run(buf364, buf362, primals_517, primals_518, primals_519, primals_520, 16384, grid=grid(16384), stream=stream0)
        del primals_520
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_521, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf366 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf367 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf367, buf365, primals_522, primals_523, primals_524, primals_525, 8192, grid=grid(8192), stream=stream0)
        del primals_525
        # Topologically Sorted Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf369 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [x_337, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38.run(buf370, buf368, primals_527, primals_528, primals_529, primals_530, 16384, grid=grid(16384), stream=stream0)
        del primals_530
        # Topologically Sorted Source Nodes: [x_339], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf372 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        buf373 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_37.run(buf373, buf371, primals_532, primals_533, primals_534, primals_535, 8192, grid=grid(8192), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf375 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf376 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_38.run(buf376, buf374, primals_537, primals_538, primals_539, primals_540, 16384, grid=grid(16384), stream=stream0)
        del primals_540
        # Topologically Sorted Source Nodes: [x_345], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_541, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 255, 2, 2), (1020, 1, 510, 255))
        buf378 = empty_strided_cuda((4, 255, 2, 2), (1020, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_345], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf377, primals_542, buf378, 1020, 4, grid=grid(1020, 4), stream=stream0)
        del buf377
        del primals_542
    return (buf332, buf355, buf378, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, buf3, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, buf4, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, buf5, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, buf6, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, buf7, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, buf8, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, buf9, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, buf10, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, buf11, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, buf12, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, buf13, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, buf14, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, buf15, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, buf16, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, buf17, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, buf18, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, buf19, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, buf20, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, buf21, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, buf22, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, buf23, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, buf24, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, buf25, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, buf26, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, buf27, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, buf28, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, buf29, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, buf30, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, buf31, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, buf32, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, buf33, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, buf34, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, buf35, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, buf36, primals_463, primals_464, primals_465, primals_467, buf37, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, buf38, primals_480, primals_481, primals_482, primals_484, primals_485, primals_486, primals_487, buf39, primals_490, primals_491, primals_492, primals_494, primals_495, primals_496, primals_497, buf40, primals_500, primals_501, primals_502, primals_504, buf41, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, buf42, primals_517, primals_518, primals_519, primals_521, primals_522, primals_523, primals_524, buf43, primals_527, primals_528, primals_529, primals_531, primals_532, primals_533, primals_534, buf44, primals_537, primals_538, primals_539, primals_541, buf45, buf47, buf48, buf50, buf51, buf53, buf55, buf56, buf58, buf59, buf61, buf62, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf84, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf101, buf102, buf104, buf105, buf107, buf108, buf110, buf111, buf113, buf114, buf116, buf117, buf119, buf120, buf122, buf123, buf125, buf126, buf128, buf129, buf131, buf132, buf134, buf135, buf137, buf138, buf140, buf141, buf143, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf155, buf156, buf158, buf159, buf161, buf163, buf164, buf166, buf167, buf169, buf170, buf172, buf173, buf175, buf176, buf178, buf179, buf181, buf182, buf184, buf185, buf187, buf188, buf190, buf191, buf193, buf194, buf196, buf197, buf199, buf200, buf202, buf203, buf205, buf206, buf208, buf209, buf211, buf212, buf214, buf215, buf217, buf218, buf220, buf221, buf223, buf225, buf226, buf228, buf229, buf231, buf232, buf234, buf235, buf237, buf238, buf240, buf241, buf243, buf244, buf246, buf247, buf249, buf250, buf252, buf253, buf255, buf256, buf258, buf259, buf261, buf262, buf264, buf266, buf269, buf272, buf276, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf288, buf289, buf291, buf292, buf294, buf295, buf297, buf298, buf300, buf301, buf303, buf304, buf306, buf307, buf309, buf310, buf312, buf313, buf315, buf316, buf318, buf319, buf321, buf322, buf324, buf325, buf327, buf328, buf330, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf344, buf345, buf347, buf348, buf350, buf351, buf353, buf356, buf358, buf359, buf361, buf362, buf364, buf365, buf367, buf368, buf370, buf371, buf373, buf374, buf376, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((255, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((255, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((255, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((255, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((255, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((255, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
