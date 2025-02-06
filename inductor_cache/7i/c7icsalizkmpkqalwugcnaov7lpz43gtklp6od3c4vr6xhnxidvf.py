# AOT ID: ['23_forward']
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


# kernel path: inductor_cache/kj/ckjif6retnxnuvnck54sexulvfnrmjoaiqswc4kbepbrtcfu5hzx.py
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
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/jz/cjz53ffn7rbc5prou4untnamayyrb5xth5jriihrn4musammwfci.py
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: inductor_cache/4m/c4mp3665u7czjb7ncuimqxmjzhjizai6k62kspkhniwikt6uo2zs.py
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
    size_hints={'y': 256, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 48*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/aa/caaphy6jmftxbj3ckdql3butbwcqp5ayaf2ivjs3siifg2jq4ar4.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/br/cbrwv5umvrvbdseg557gnqkloomga3mn5bs2bii3i4wgunko3ic2.py
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
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 96*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bu/cbuy56fby4itmxwcn5644vhxoovupz2o7uniixicmdxc743yz2lc.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/li/cliix3fgqukdcgek46ydx5hnutbkhjbcjuu4rlylyiptblkkbg4n.py
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
    size_hints={'y': 4096, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 3
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 192*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pj/cpjl5o5fbsjplkq3v7xaazivrxijjei4sbivfyw6x3osip7lurem.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5k/c5ksaljktaerpjvvpke2tcrfc2ejg5irgslheno75blifxto77td.py
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
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 3200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3dnfpbcc34gzzjesg5zfhhf6fc3wztxbiu6a7fpdfu6easmgd7.py
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
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 6272*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wt/cwt6aynfm6wjrahid2jra5pfcpm2mptwb44tyrh4j6ypo6ucrwwv.py
# Topologically Sorted Source Nodes: [x1_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x1_1 => getitem_1
# Graph fragment:
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_10 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 32)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 528*x1 + 17424*x2), None)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 528*x1 + 17424*x2), None)
    tmp7 = tl.load(in_ptr0 + (528 + x0 + 528*x1 + 17424*x2), None)
    tmp12 = tl.load(in_ptr0 + (544 + x0 + 528*x1 + 17424*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/5z/c5zrwi77fvdfxzcogi622fiqxc3jy56wonmwttm46eyvjeoib4xk.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %getitem_2], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 32)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*x0 + 528*x1 + 17424*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (16 + 16*x0 + 528*x1 + 17424*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (528 + 16*x0 + 528*x1 + 17424*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (544 + 16*x0 + 528*x1 + 17424*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 32, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (16*x0 + 528*x1 + 17424*x3 + ((-16) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr1 + (16 + 16*x0 + 528*x1 + 17424*x3 + ((-16) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.load(in_ptr1 + (528 + 16*x0 + 528*x1 + 17424*x3 + ((-16) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp22 = tl.load(in_ptr1 + (544 + 16*x0 + 528*x1 + 17424*x3 + ((-16) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqx6vqss3ozoixidcwtdx4kod43kohn6w2tsjrnxgycqz2si6g37.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_4, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y0 + 32768*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 16384*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/6m/c6m2aqcgu5ddzgrzqjs366uvm7k6zmisrifipylb4wsfth2yiqgs.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_3 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_relu_13 = async_compile.triton('triton_poi_fused_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_13(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciifscrgpqnrzghfilmegptpeapeos77sqlkjdpxm2l7jyovtsjo.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_1, mul_1, mul_2, sub
#   input_6 => relu_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/nd/cndj664i234cahy2yzmho2b35p3ci65mznmmp6zy4lcgd76wxma6.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_12 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_5, %primals_16, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (16384 + x2 + 1024*y0 + 32768*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 16384*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxsxi5yz4juabzew255jqk57cnbpzjurfuvslbkklfg23a57tfp.py
# Topologically Sorted Source Nodes: [out, add, out_1, x_1], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
# Source node to ATen node mapping:
#   add => add_8
#   out => cat_1
#   out_1 => relu_8
#   x_1 => clone
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_3, %relu_7], 1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %cat), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %le_116 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
triton_poi_fused_add_cat_clone_relu_threshold_backward_16 = async_compile.triton('triton_poi_fused_add_cat_clone_relu_threshold_backward_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_clone_relu_threshold_backward_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_clone_relu_threshold_backward_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 32)
    x2 = xindex
    y1 = yindex // 32
    y5 = yindex
    y3 = (yindex % 16)
    y4 = ((yindex // 16) % 2)
    tmp49 = tl.load(in_ptr10 + (x2 + 1024*y5), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*x2 + 16384*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1, 1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 32, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (16*x2 + 16384*y1 + ((-16) + y0)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + (tl.broadcast_to((-16) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + (tl.broadcast_to((-16) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1, 1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + (tl.broadcast_to((-16) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + (tl.broadcast_to((-16) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1, 1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tmp50 = tmp48 + tmp49
    tmp51 = tl.full([1, 1], 0, tl.int32)
    tmp52 = triton_helpers.maximum(tmp51, tmp50)
    tmp53 = 0.0
    tmp54 = tmp52 <= tmp53
    tl.store(out_ptr1 + (x2 + 1024*y4 + 2048*y3 + 32768*y1), tmp52, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 32*x2 + 32768*y1), tmp54, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ko/ckoyg7enddsizc6siluhkxjycnjzv5zdk4qbstvfzbcvy4dqnll4.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_7 => clone_2
#   x_8 => view_8
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [4, -1, 32, 32]), kwargs = {})
triton_poi_fused_clone_view_17 = async_compile.triton('triton_poi_fused_clone_view_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 32768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xf/cxfewezbjbeq7rcktz4zyrglg5dwmtcustj73illu35wkialv5oe.py
# Topologically Sorted Source Nodes: [x1_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x1_6 => getitem_11
# Graph fragment:
#   %getitem_11 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_18 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 16)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 544*x1 + 9248*x2), None)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + 544*x1 + 9248*x2), None)
    tmp7 = tl.load(in_ptr0 + (544 + x0 + 544*x1 + 9248*x2), None)
    tmp12 = tl.load(in_ptr0 + (576 + x0 + 544*x1 + 9248*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikuaqr545awibf242gxjzdhf3ejadkcojjdccefvzeorpj464zj.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_62 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_10, %getitem_12], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 64)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x3 = xindex // 16384
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x0 + 544*x1 + 9248*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (32 + 32*x0 + 544*x1 + 9248*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (544 + 32*x0 + 544*x1 + 9248*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (576 + 32*x0 + 544*x1 + 9248*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 64, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (32*x0 + 544*x1 + 9248*x3 + ((-32) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr1 + (32 + 32*x0 + 544*x1 + 9248*x3 + ((-32) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.load(in_ptr1 + (544 + 32*x0 + 544*x1 + 9248*x3 + ((-32) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp22 = tl.load(in_ptr1 + (576 + 32*x0 + 544*x1 + 9248*x3 + ((-32) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckmks55wveu2o7vhygdldqfnviwwkaey6fdracqcrrt7ojzmpxnz.py
# Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_63 => convolution_28
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %primals_78, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y0 + 16384*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 8192*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/iq/ciqwa7r2n6zn4lgsn2xwhn3uigg3wfauvdzpn76sqveswunn3dj4.py
# Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_64 => relu_27
# Graph fragment:
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
triton_poi_fused_relu_21 = async_compile.triton('triton_poi_fused_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_21(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/v2/cv2uwg2zm4eut5kqujlvg3w6qmrzln7brrhqqkucqwmq5p3yubue.py
# Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_66 => add_28, mul_37, mul_38, sub_12
#   input_67 => relu_28
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhenwv5m3yl65u4dcdsy6oa74eys5ygbwpc4dsib2rjgjhqp3uo.py
# Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_73 => convolution_32
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_15, %primals_90, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_23 = async_compile.triton('triton_poi_fused_convolution_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (8192 + x2 + 256*y0 + 16384*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 8192*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3n/c3n5bdhyjvvhgnljbm753cgh52zhplljenxmbbyykjixhdc2cc2a.py
# Topologically Sorted Source Nodes: [out_6, add_3, out_7, x_10], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
# Source node to ATen node mapping:
#   add_3 => add_35
#   out_6 => cat_5
#   out_7 => relu_35
#   x_10 => clone_3
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_30, %relu_34], 1), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_5, %cat_4), kwargs = {})
#   %relu_35 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %le_89 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_35, 0), kwargs = {})
triton_poi_fused_add_cat_clone_relu_threshold_backward_24 = async_compile.triton('triton_poi_fused_add_cat_clone_relu_threshold_backward_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_clone_relu_threshold_backward_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_clone_relu_threshold_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 64)
    x2 = xindex
    y1 = yindex // 64
    y5 = yindex
    y3 = (yindex % 32)
    y4 = ((yindex // 32) % 2)
    tmp49 = tl.load(in_ptr10 + (x2 + 256*y5), xmask & ymask)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x2 + 8192*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1, 1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 64, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (32*x2 + 8192*y1 + ((-32) + y0)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + (tl.broadcast_to((-32) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + (tl.broadcast_to((-32) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1, 1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + (tl.broadcast_to((-32) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + (tl.broadcast_to((-32) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1, 1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tmp50 = tmp48 + tmp49
    tmp51 = tl.full([1, 1], 0, tl.int32)
    tmp52 = triton_helpers.maximum(tmp51, tmp50)
    tmp53 = 0.0
    tmp54 = tmp52 <= tmp53
    tl.store(out_ptr1 + (x2 + 256*y4 + 512*y3 + 16384*y1), tmp52, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 64*x2 + 16384*y1), tmp54, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjingn5efg3a3hlsbr755tfxu4muzd6s6heqearptqqdd5eyms32.py
# Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_13 => clone_4
#   x_14 => view_14
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [4, -1, 16, 16]), kwargs = {})
triton_poi_fused_clone_view_25 = async_compile.triton('triton_poi_fused_clone_view_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 64*x2 + 16384*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xx/cxxkcn7afbsm7zcrbjifi6uxhyj3upcjhavjtbja76pej25qk7bi.py
# Topologically Sorted Source Nodes: [x1_10], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x1_10 => getitem_19
# Graph fragment:
#   %getitem_19 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_26 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 8)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576*x1 + 5184*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 576*x1 + 5184*x2), None)
    tmp7 = tl.load(in_ptr0 + (576 + x0 + 576*x1 + 5184*x2), None)
    tmp12 = tl.load(in_ptr0 + (640 + x0 + 576*x1 + 5184*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4iwqrcaa5nt7cmlb734o2j7xvhtz5o6xvgqflpnvcfgxinb6axm.py
# Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_103 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_18, %getitem_20], 1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 128)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x3 = xindex // 8192
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x0 + 576*x1 + 5184*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (64 + 64*x0 + 576*x1 + 5184*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (576 + 64*x0 + 576*x1 + 5184*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (640 + 64*x0 + 576*x1 + 5184*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 128, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (64*x0 + 576*x1 + 5184*x3 + ((-64) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr1 + (64 + 64*x0 + 576*x1 + 5184*x3 + ((-64) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.load(in_ptr1 + (576 + 64*x0 + 576*x1 + 5184*x3 + ((-64) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp22 = tl.load(in_ptr1 + (640 + 64*x0 + 576*x1 + 5184*x3 + ((-64) + x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, None)
''', device_str='cuda')


# kernel path: inductor_cache/qj/cqjbpg7qkjtiye6fqkhayinnacop55zxtt2igzgjq3gr32zu72iy.py
# Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_104 => convolution_46
# Graph fragment:
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_22, %primals_128, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y0 + 8192*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 4096*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/oy/coyp7ftsow6pvp3wqwd6ywvvadwfmwe6fipqp4knbgvffp5tckpj.py
# Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_105 => relu_45
# Graph fragment:
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_46,), kwargs = {})
triton_poi_fused_relu_29 = async_compile.triton('triton_poi_fused_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_29(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ej/cejdhouwlrrn3ubuyw6jdihbttuxi5appkdpltloye7hram2kxge.py
# Topologically Sorted Source Nodes: [input_107, input_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_107 => add_46, mul_61, mul_62, sub_20
#   input_108 => relu_46
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_47, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_46 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_46,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rkmrynk5q7fvzpotwcvmbzaxgzah3raaiipdi37qeg5dksukdh.py
# Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_114 => convolution_50
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_23, %primals_140, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (4096 + x2 + 64*y0 + 8192*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 4096*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cl/cclf6fdsfcrwu3ednodpbznmvyyu6wqyv2prfe7bbqfdrpfnb64y.py
# Topologically Sorted Source Nodes: [out_10, add_5, out_11, x_16], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
# Source node to ATen node mapping:
#   add_5 => add_53
#   out_10 => cat_8
#   out_11 => relu_53
#   x_16 => clone_5
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_48, %relu_52], 1), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_8, %cat_7), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_11,), kwargs = {memory_format: torch.contiguous_format})
#   %le_71 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_53, 0), kwargs = {})
triton_poi_fused_add_cat_clone_relu_threshold_backward_32 = async_compile.triton('triton_poi_fused_add_cat_clone_relu_threshold_backward_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_clone_relu_threshold_backward_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_clone_relu_threshold_backward_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 128)
    x2 = xindex
    y1 = yindex // 128
    y5 = yindex
    y3 = (yindex % 64)
    y4 = ((yindex // 64) % 2)
    tmp49 = tl.load(in_ptr10 + (x2 + 64*y5), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x2 + 4096*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1, 1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (64*x2 + 4096*y1 + ((-64) + y0)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1, 1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1, 1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tmp50 = tmp48 + tmp49
    tmp51 = tl.full([1, 1], 0, tl.int32)
    tmp52 = triton_helpers.maximum(tmp51, tmp50)
    tmp53 = 0.0
    tmp54 = tmp52 <= tmp53
    tl.store(out_ptr1 + (x2 + 64*y4 + 128*y3 + 8192*y1), tmp52, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 128*x2 + 8192*y1), tmp54, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/er/cerznj6yrsvfgv5jrh6kaww6ufm5dlap5ncgdd2fy2kac6edyxu7.py
# Topologically Sorted Source Nodes: [x_37, x_38, input_264], Original ATen: [aten.clone, aten.view, aten.mean]
# Source node to ATen node mapping:
#   input_264 => mean
#   x_37 => clone_12
#   x_38 => view_38
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
#   %view_38 : [num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [4, -1, 8, 8]), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_38, [-1, -2], True), kwargs = {})
triton_per_fused_clone_mean_view_33 = async_compile.triton('triton_per_fused_clone_mean_view_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_mean_view_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_mean_view_33(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + (x0 + 128*r2 + 8192*x1), tmp0, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2kvcfox36wzggu2q5rqsaa5yul3y3w6moi4hw4pqrqh2jplmae.py
# Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_40 => add_118, mul_157, mul_158, sub_52
#   x_41 => relu_117
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_417), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_421), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_423), kwargs = {})
#   %relu_117 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_118,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/mw/cmwx7ei5lk2fbvujevo6hjbyhljtlepde4yvs72hwfyfem4twzuy.py
# Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_43 => add_120, mul_160, mul_161, sub_53
#   x_44 => relu_118
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_118 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_120,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wg/cwgzhkoajhfbictkxfpjdsmse7ugxmtvhrj235heifhy4f3ptmvl.py
# Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_46 => add_122, mul_163, mul_164, sub_54
#   x_47 => relu_119
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_433), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_437), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_439), kwargs = {})
#   %relu_119 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_122,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gz/cgzg6vbp3cs45wx5huf64pj4cwss2hu7fon3zd6dpt6fq2ofvuxp.py
# Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_49 => add_124, mul_166, mul_167, sub_55
#   x_50 => relu_120
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_120 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_124,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oc/cocaqic4borl7lvufznblrwzvqen4p4rt6vpdhry5hg7sh2qfcco.py
# Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_26 => convert_element_type_113
# Graph fragment:
#   %convert_element_type_113 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_39, torch.int64), kwargs = {})
triton_poi_fused__to_copy_38 = async_compile.triton('triton_poi_fused__to_copy_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_38(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sl/csllkdyflvrkv7hxrql2h4kdplwezedbofwsxpgjghf6qukvhnyv.py
# Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   out_26 => clamp_max_2, clamp_min, clamp_min_2, convert_element_type_112, iota, mul_168, sub_56
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_112, 0.0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_168, 0.0), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_115), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_56, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_39 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_39(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x3/cx3bsplwxnra7k2uxjnrxn72uuxz347jv3tpy44rlpra3xdltfwl.py
# Topologically Sorted Source Nodes: [out_26, x_52, x_53, out_27], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_26 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_127, add_128, add_129, mul_170, mul_171, mul_172, sub_57, sub_58, sub_60
#   out_27 => add_132
#   x_52 => add_131, mul_174, mul_175, sub_61
#   x_53 => relu_121
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_120, [None, None, %convert_element_type_113, %convert_element_type_115]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_120, [None, None, %convert_element_type_113, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_120, [None, None, %clamp_max, %convert_element_type_115]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_120, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %clamp_max_2), kwargs = {})
#   %add_127 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_170), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %clamp_max_2), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_171), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_128, %add_127), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %clamp_max_3), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %mul_172), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_449), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_451), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_174, %unsqueeze_453), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_175, %unsqueeze_455), kwargs = {})
#   %relu_121 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
#   %add_132 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_121, %add_129), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 2
    x1 = (xindex % 2)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 4)
    y4 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (y3 + 4*x5 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y3), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (y3), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (y3), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (y3), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1, 1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp37 = tmp36 + tmp1
    tmp38 = tmp36 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp36)
    tmp40 = tmp17 - tmp17
    tmp42 = tmp40 * tmp41
    tmp43 = tmp17 + tmp42
    tmp44 = tmp35 + tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + 4*y0), tmp44, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kp/ckp2idd724huqch6x23vgd4gkqohbjiv52v7yiiyk7tobwu3ixyl.py
# Topologically Sorted Source Nodes: [out_28], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_28 => convert_element_type_119
# Graph fragment:
#   %convert_element_type_119 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_41, torch.int64), kwargs = {})
triton_poi_fused__to_copy_41 = async_compile.triton('triton_poi_fused__to_copy_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd2gzkmxqua4oyj3gdy2buv3qratcupw53xzacg6onj5jz72iomr.py
# Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   out_28 => add_133, clamp_max_4
# Graph fragment:
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_119, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_133, 1), kwargs = {})
triton_poi_fused_add_clamp_42 = async_compile.triton('triton_poi_fused_add_clamp_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bu/cbub5fwcm3bbo6bvx3mxri67nbmpkkjhalbvx7txhzmlu3zvlckz.py
# Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   out_28 => clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_118, iota_2, mul_176, sub_62
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_118 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_118, 0.3333333333333333), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_176, 0.0), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_121), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_62, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_43 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_43', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_43(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kr/ckrelcujrt46hmmuitw5a5537pqwvf5scy33zy62ux73pjmgpy5w.py
# Topologically Sorted Source Nodes: [out_28, x_55, x_56, out_29], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_28 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_135, add_136, add_137, mul_178, mul_179, mul_180, sub_63, sub_64, sub_66
#   out_29 => add_140
#   x_55 => add_139, mul_182, mul_183, sub_67
#   x_56 => relu_122
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_132, [None, None, %convert_element_type_119, %convert_element_type_121]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_132, [None, None, %convert_element_type_119, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_132, [None, None, %clamp_max_4, %convert_element_type_121]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_132, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %clamp_max_6), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_178), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %clamp_max_6), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_179), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_136, %add_135), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %clamp_max_7), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %mul_180), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_457), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_459), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_461), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %unsqueeze_463), kwargs = {})
#   %relu_122 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_139,), kwargs = {})
#   %add_140 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_122, %add_137), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 4
    x1 = (xindex % 4)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 4)
    y4 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (y3 + 4*x5 + 64*y4), xmask & ymask)
    tmp20 = tl.load(in_ptr6 + (y3), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (y3), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y3), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (y3), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1, 1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1, 1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp38 = tmp37 + tmp1
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr2 + (tmp8 + 2*tmp40 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (tmp13 + 2*tmp40 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp16
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp48 = tmp46 * tmp47
    tmp49 = tmp18 + tmp48
    tmp50 = tmp36 + tmp49
    tl.store(in_out_ptr0 + (x5 + 16*y0), tmp50, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/iz/ciz2ndntgjdhmjhmwktcm5epcupvhjdawq3wu7sg3rhtblfbe7jr.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_30 => convert_element_type_125
# Graph fragment:
#   %convert_element_type_125 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_43, torch.int64), kwargs = {})
triton_poi_fused__to_copy_45 = async_compile.triton('triton_poi_fused__to_copy_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_45(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3ee2uu72ssrx4cguq5d5uiuodexr6xyq4sjv3qq47lsig4eocu7.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   out_30 => add_141, clamp_max_8
# Graph fragment:
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_125, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_141, 3), kwargs = {})
triton_poi_fused_add_clamp_46 = async_compile.triton('triton_poi_fused_add_clamp_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pp/cpprtnomakfm4mcxonxzacq62ltu3ntq2tejyfcmknirxazwl7rp.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   out_30 => clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_124, iota_4, mul_184, sub_68
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_124, 0.42857142857142855), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_184, 0.0), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_127), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_68, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_47 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_47', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbr5wik46bu77lp74unegitkntspjmgac5rkji7mpr3fs42s3d2b.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   out_30 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_143, add_144, add_145, mul_186, mul_187, mul_188, sub_69, sub_70, sub_72
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_140, [None, None, %convert_element_type_125, %convert_element_type_127]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_140, [None, None, %convert_element_type_125, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_140, [None, None, %clamp_max_8, %convert_element_type_127]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_140, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %clamp_max_10), kwargs = {})
#   %add_143 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_186), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %clamp_max_10), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_187), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_144, %add_143), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %clamp_max_11), kwargs = {})
#   %add_145 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_143, %mul_188), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_48 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 8
    x1 = (xindex % 8)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 4)
    y4 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 4*tmp22 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 4*tmp22 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tl.store(out_ptr1 + (y3 + 4*x5 + 256*y4), tmp31, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/g3/cg3taqgyuw25my4r35566dlqvkdlkovwtmvklgm6len7tzx67ayp.py
# Topologically Sorted Source Nodes: [x_58, x_59, out_31, x_61, x_62, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.add]
# Source node to ATen node mapping:
#   out_31 => mul_192
#   out_32 => add_150
#   x_58 => add_147, mul_190, mul_191, sub_73
#   x_59 => relu_123
#   x_61 => add_149, mul_194, mul_195, sub_74
#   x_62 => relu_124
# Graph fragment:
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_465), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_467), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_469), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_471), kwargs = {})
#   %relu_123 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_123, %add_145), kwargs = {})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_473), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_475), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_194, %unsqueeze_477), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %unsqueeze_479), kwargs = {})
#   %relu_124 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_149,), kwargs = {})
#   %add_150 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_124, %mul_192), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x3), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp0 + tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nx/cnx4yjp43qdxt3ts3shumbxtniosmom2bxx2idcqn4xhdr3sarbj.py
# Topologically Sorted Source Nodes: [x_63], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_63 => convert_element_type_133
# Graph fragment:
#   %convert_element_type_133 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_45, torch.int64), kwargs = {})
triton_poi_fused__to_copy_50 = async_compile.triton('triton_poi_fused__to_copy_50', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_50(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.1111111111111111
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p7/cp74hgquua6hvj25uhtk3mxlscpmykajiwxym3k6bbayvvoqt5yb.py
# Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_63 => add_151, clamp_max_12
# Graph fragment:
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_133, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_151, 7), kwargs = {})
triton_poi_fused_add_clamp_51 = async_compile.triton('triton_poi_fused_add_clamp_51', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.1111111111111111
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 7, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ah/cah44tzp5yescdn2sztt37w5jlq4qzzrjmqum47ccd3pnoe3ebem.py
# Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   x_63 => clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_132, iota_6, mul_196, sub_75
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_132, 0.1111111111111111), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_196, 0.0), kwargs = {})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_135), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_75, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_52 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_52', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_52(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.1111111111111111
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w6/cw6sh2xlamegk4o7ollm4sxcdm5r7zeke4cglbpp6uii56noty3o.py
# Topologically Sorted Source Nodes: [x_63], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_63 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_153, add_154, add_155, mul_198, mul_199, mul_200, sub_76, sub_77, sub_79
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_150, [None, None, %convert_element_type_133, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_150, [None, None, %convert_element_type_133, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_150, [None, None, %clamp_max_12, %convert_element_type_135]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_150, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %clamp_max_14), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_198), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %clamp_max_14), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_199), kwargs = {})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_154, %add_153), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %clamp_max_15), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %mul_200), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_53 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = ((xindex // 4096) % 4)
    x3 = xindex // 16384
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 4*tmp8 + 32*tmp4 + 256*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (x2 + 4*tmp13 + 32*tmp4 + 256*x3), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (x2 + 4*tmp8 + 32*tmp22 + 256*x3), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x2 + 4*tmp13 + 32*tmp22 + 256*x3), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tl.store(in_out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_4, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_5, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_11, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_12, (16, ), (1, ))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_17, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (16, ), (1, ))
    assert_size_stride(primals_22, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_23, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_24, (16, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (16, ), (1, ))
    assert_size_stride(primals_27, (16, ), (1, ))
    assert_size_stride(primals_28, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_29, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (16, ), (1, ))
    assert_size_stride(primals_32, (16, ), (1, ))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_35, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (16, ), (1, ))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_41, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_42, (16, ), (1, ))
    assert_size_stride(primals_43, (16, ), (1, ))
    assert_size_stride(primals_44, (16, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_47, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (16, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (16, ), (1, ))
    assert_size_stride(primals_52, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_53, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_54, (16, ), (1, ))
    assert_size_stride(primals_55, (16, ), (1, ))
    assert_size_stride(primals_56, (16, ), (1, ))
    assert_size_stride(primals_57, (16, ), (1, ))
    assert_size_stride(primals_58, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_59, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_60, (16, ), (1, ))
    assert_size_stride(primals_61, (16, ), (1, ))
    assert_size_stride(primals_62, (16, ), (1, ))
    assert_size_stride(primals_63, (16, ), (1, ))
    assert_size_stride(primals_64, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_65, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_66, (16, ), (1, ))
    assert_size_stride(primals_67, (16, ), (1, ))
    assert_size_stride(primals_68, (16, ), (1, ))
    assert_size_stride(primals_69, (16, ), (1, ))
    assert_size_stride(primals_70, (16, 16, 1, 3), (48, 3, 3, 1))
    assert_size_stride(primals_71, (16, 16, 3, 1), (48, 3, 1, 1))
    assert_size_stride(primals_72, (16, ), (1, ))
    assert_size_stride(primals_73, (16, ), (1, ))
    assert_size_stride(primals_74, (16, ), (1, ))
    assert_size_stride(primals_75, (16, ), (1, ))
    assert_size_stride(primals_76, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_77, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_78, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_79, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (32, ), (1, ))
    assert_size_stride(primals_83, (32, ), (1, ))
    assert_size_stride(primals_84, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_85, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_86, (32, ), (1, ))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_91, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_92, (32, ), (1, ))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_97, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_103, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_104, (32, ), (1, ))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_109, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_110, (32, ), (1, ))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (32, ), (1, ))
    assert_size_stride(primals_114, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_115, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_116, (32, ), (1, ))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, 32, 1, 3), (96, 3, 3, 1))
    assert_size_stride(primals_121, (32, 32, 3, 1), (96, 3, 1, 1))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (32, ), (1, ))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_127, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_128, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_129, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (64, ), (1, ))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_135, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_141, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_142, (64, ), (1, ))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_147, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_153, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, ), (1, ))
    assert_size_stride(primals_158, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_159, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_160, (64, ), (1, ))
    assert_size_stride(primals_161, (64, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_165, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (64, ), (1, ))
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_171, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_177, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_183, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_189, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, ), (1, ))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_195, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_201, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_202, (64, ), (1, ))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_207, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_213, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_214, (64, ), (1, ))
    assert_size_stride(primals_215, (64, ), (1, ))
    assert_size_stride(primals_216, (64, ), (1, ))
    assert_size_stride(primals_217, (64, ), (1, ))
    assert_size_stride(primals_218, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_219, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (64, ), (1, ))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_225, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_226, (64, ), (1, ))
    assert_size_stride(primals_227, (64, ), (1, ))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_231, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_232, (64, ), (1, ))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (64, ), (1, ))
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_237, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, ), (1, ))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_243, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, ), (1, ))
    assert_size_stride(primals_248, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_249, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_255, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, ), (1, ))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_261, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_262, (64, ), (1, ))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, ), (1, ))
    assert_size_stride(primals_265, (64, ), (1, ))
    assert_size_stride(primals_266, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_267, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_268, (64, ), (1, ))
    assert_size_stride(primals_269, (64, ), (1, ))
    assert_size_stride(primals_270, (64, ), (1, ))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_273, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_274, (64, ), (1, ))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, ), (1, ))
    assert_size_stride(primals_277, (64, ), (1, ))
    assert_size_stride(primals_278, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_279, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, ), (1, ))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_285, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_286, (64, ), (1, ))
    assert_size_stride(primals_287, (64, ), (1, ))
    assert_size_stride(primals_288, (64, ), (1, ))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_291, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (64, ), (1, ))
    assert_size_stride(primals_295, (64, ), (1, ))
    assert_size_stride(primals_296, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_297, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_298, (64, ), (1, ))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_303, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_304, (64, ), (1, ))
    assert_size_stride(primals_305, (64, ), (1, ))
    assert_size_stride(primals_306, (64, ), (1, ))
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_309, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (64, ), (1, ))
    assert_size_stride(primals_312, (64, ), (1, ))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, 64, 1, 3), (192, 3, 3, 1))
    assert_size_stride(primals_315, (64, 64, 3, 1), (192, 3, 1, 1))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (64, ), (1, ))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_321, (128, ), (1, ))
    assert_size_stride(primals_322, (128, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_326, (128, ), (1, ))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, 128, 7, 7), (6272, 49, 7, 1))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_336, (4, ), (1, ))
    assert_size_stride(primals_337, (4, ), (1, ))
    assert_size_stride(primals_338, (4, ), (1, ))
    assert_size_stride(primals_339, (4, ), (1, ))
    assert_size_stride(primals_340, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_341, (4, ), (1, ))
    assert_size_stride(primals_342, (4, ), (1, ))
    assert_size_stride(primals_343, (4, ), (1, ))
    assert_size_stride(primals_344, (4, ), (1, ))
    assert_size_stride(primals_345, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_346, (4, ), (1, ))
    assert_size_stride(primals_347, (4, ), (1, ))
    assert_size_stride(primals_348, (4, ), (1, ))
    assert_size_stride(primals_349, (4, ), (1, ))
    assert_size_stride(primals_350, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_351, (4, ), (1, ))
    assert_size_stride(primals_352, (4, ), (1, ))
    assert_size_stride(primals_353, (4, ), (1, ))
    assert_size_stride(primals_354, (4, ), (1, ))
    assert_size_stride(primals_355, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_356, (4, ), (1, ))
    assert_size_stride(primals_357, (4, ), (1, ))
    assert_size_stride(primals_358, (4, ), (1, ))
    assert_size_stride(primals_359, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_3, buf2, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_3
        buf3 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_4, buf3, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_4
        buf4 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_5, buf4, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_5
        buf5 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_10, buf5, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_10
        buf6 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_11, buf6, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_11
        buf7 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_16, buf7, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_16
        buf8 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf8, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_17
        buf9 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_22, buf9, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_22
        buf10 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_23, buf10, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_23
        buf11 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_28, buf11, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_28
        buf12 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_29, buf12, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_29
        buf13 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_34, buf13, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_34
        buf14 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_35, buf14, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_35
        buf15 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_40, buf15, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_40
        buf16 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_41, buf16, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_41
        buf17 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_46, buf17, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_46
        buf18 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_47, buf18, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_47
        buf19 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_52, buf19, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_52
        buf20 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_53, buf20, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_53
        buf21 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_58, buf21, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_58
        buf22 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_59, buf22, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_59
        buf23 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_64, buf23, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_64
        buf24 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_65, buf24, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_65
        buf25 = empty_strided_cuda((16, 16, 1, 3), (48, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_70, buf25, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_70
        buf26 = empty_strided_cuda((16, 16, 3, 1), (48, 1, 16, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_71, buf26, 256, 3, grid=grid(256, 3), stream=stream0)
        del primals_71
        buf27 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_76, buf27, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_76
        buf28 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_77, buf28, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_77
        buf29 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_78, buf29, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_78
        buf30 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_79, buf30, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_79
        buf31 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_84, buf31, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_84
        buf32 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_85, buf32, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_85
        buf33 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_90, buf33, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_90
        buf34 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_91, buf34, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_91
        buf35 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_96, buf35, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_96
        buf36 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_97, buf36, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_97
        buf37 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_102, buf37, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_102
        buf38 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_103, buf38, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_103
        buf39 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_108, buf39, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_108
        buf40 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_109, buf40, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_109
        buf41 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_114, buf41, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_114
        buf42 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_115, buf42, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_115
        buf43 = empty_strided_cuda((32, 32, 1, 3), (96, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_120, buf43, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_120
        buf44 = empty_strided_cuda((32, 32, 3, 1), (96, 1, 32, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_121, buf44, 1024, 3, grid=grid(1024, 3), stream=stream0)
        del primals_121
        buf45 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_126, buf45, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_126
        buf46 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_127, buf46, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_127
        buf47 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_128, buf47, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_128
        buf48 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_129, buf48, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_129
        buf49 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_134, buf49, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_134
        buf50 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_135, buf50, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_135
        buf51 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_140, buf51, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_140
        buf52 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_141, buf52, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_141
        buf53 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_146, buf53, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_146
        buf54 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_147, buf54, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_147
        buf55 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_152, buf55, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_152
        buf56 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_153, buf56, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_153
        buf57 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_158, buf57, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_158
        buf58 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_159, buf58, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_159
        buf59 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_164, buf59, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_164
        buf60 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_165, buf60, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_165
        buf61 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_170, buf61, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_170
        buf62 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_171, buf62, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_171
        buf63 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_176, buf63, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_176
        buf64 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_177, buf64, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_177
        buf65 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_182, buf65, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_182
        buf66 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_183, buf66, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_183
        buf67 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_188, buf67, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_188
        buf68 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_189, buf68, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_189
        buf69 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_194, buf69, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_194
        buf70 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_195, buf70, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_195
        buf71 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_200, buf71, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_200
        buf72 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_201, buf72, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_201
        buf73 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_206, buf73, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_206
        buf74 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_207, buf74, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_207
        buf75 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_212, buf75, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_212
        buf76 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_213, buf76, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_213
        buf77 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_218, buf77, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_218
        buf78 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_219, buf78, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_219
        buf79 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_224, buf79, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_224
        buf80 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_225, buf80, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_225
        buf81 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_230, buf81, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_230
        buf82 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_231, buf82, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_231
        buf83 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_236, buf83, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_236
        buf84 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_237, buf84, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_237
        buf85 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_242, buf85, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_242
        buf86 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_243, buf86, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_243
        buf87 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_248, buf87, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_248
        buf88 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_249, buf88, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_249
        buf89 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_254, buf89, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_254
        buf90 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_255, buf90, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_255
        buf91 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_260, buf91, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_260
        buf92 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_261, buf92, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_261
        buf93 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_266, buf93, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_266
        buf94 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_267, buf94, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_267
        buf95 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_272, buf95, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_272
        buf96 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_273, buf96, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_273
        buf97 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_278, buf97, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_278
        buf98 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_279, buf98, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_279
        buf99 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_284, buf99, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_284
        buf100 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_285, buf100, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_285
        buf101 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_290, buf101, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_290
        buf102 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_291, buf102, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_291
        buf103 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_296, buf103, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_296
        buf104 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_297, buf104, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_297
        buf105 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_302, buf105, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_302
        buf106 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_303, buf106, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_303
        buf107 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_308, buf107, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_308
        buf108 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_309, buf108, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_309
        buf109 = empty_strided_cuda((64, 64, 1, 3), (192, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_314, buf109, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_314
        buf110 = empty_strided_cuda((64, 64, 3, 1), (192, 1, 64, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_315, buf110, 4096, 3, grid=grid(4096, 3), stream=stream0)
        del primals_315
        buf111 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_320, buf111, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_320
        buf112 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_325, buf112, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_325
        buf113 = empty_strided_cuda((128, 128, 7, 7), (6272, 1, 896, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_330, buf113, 16384, 49, grid=grid(16384, 49), stream=stream0)
        del primals_330
        # Topologically Sorted Source Nodes: [x1], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 16, 33, 33), (17424, 1, 528, 16))
        buf115 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.int8)
        # Topologically Sorted Source Nodes: [x1_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_10.run(buf114, buf115, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x2], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf0, buf2, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 16, 33, 33), (17424, 1, 528, 16))
        buf117 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.int8)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_10.run(buf116, buf117, 65536, grid=grid(65536), stream=stream0)
        buf118 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf114, buf116, buf118, 131072, grid=grid(131072), stream=stream0)
        buf119 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf118, buf119, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf3, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf121, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, buf4, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf123 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf122, primals_6, primals_7, primals_8, primals_9, buf123, 65536, grid=grid(65536), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, buf5, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf125, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, buf6, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf127 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf118, buf127, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf7, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf129, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, buf8, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf131 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf130, primals_18, primals_19, primals_20, primals_21, buf131, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, buf9, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf133, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, buf10, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf136 = empty_strided_cuda((4, 16, 2, 32, 32), (32768, 2048, 1024, 32, 1), torch.float32)
        buf426 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.bool)
        # Topologically Sorted Source Nodes: [out, add, out_1, x_1], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_16.run(buf126, primals_12, primals_13, primals_14, primals_15, buf134, primals_24, primals_25, primals_26, primals_27, buf118, buf136, buf426, 128, 1024, grid=grid(128, 1024), stream=stream0)
        buf137 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf136, buf137, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf11, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf139, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, buf12, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf141 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf140, primals_30, primals_31, primals_32, primals_33, buf141, 65536, grid=grid(65536), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, buf13, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf143, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, buf14, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf145 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf136, buf145, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, buf15, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf147, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, buf16, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf149 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf148, primals_42, primals_43, primals_44, primals_45, buf149, 65536, grid=grid(65536), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, buf17, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf151, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, buf18, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf154 = empty_strided_cuda((4, 16, 2, 32, 32), (32768, 2048, 1024, 32, 1), torch.float32)
        buf425 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.bool)
        # Topologically Sorted Source Nodes: [out_2, add_1, out_3, x_4], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_16.run(buf144, primals_36, primals_37, primals_38, primals_39, buf152, primals_48, primals_49, primals_50, primals_51, buf136, buf154, buf425, 128, 1024, grid=grid(128, 1024), stream=stream0)
        buf155 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf154, buf155, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, buf19, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf157, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf20, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf159 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf158, primals_54, primals_55, primals_56, primals_57, buf159, 65536, grid=grid(65536), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, buf21, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf161, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, buf22, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf163 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf154, buf163, 64, 1024, grid=grid(64, 1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, buf23, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf165, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, buf24, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf167 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [input_55, input_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf166, primals_66, primals_67, primals_68, primals_69, buf167, 65536, grid=grid(65536), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, buf25, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_13.run(buf169, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf26, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf172 = empty_strided_cuda((4, 16, 2, 32, 32), (32768, 2048, 1024, 32, 1), torch.float32)
        buf424 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.bool)
        # Topologically Sorted Source Nodes: [out_4, add_2, out_5, x_7], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_16.run(buf162, primals_60, primals_61, primals_62, primals_63, buf170, primals_72, primals_73, primals_74, primals_75, buf154, buf172, buf424, 128, 1024, grid=grid(128, 1024), stream=stream0)
        buf173 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_17.run(buf172, buf173, 128, 1024, grid=grid(128, 1024), stream=stream0)
        del buf172
        # Topologically Sorted Source Nodes: [x1_5], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, buf27, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 32, 17, 17), (9248, 1, 544, 32))
        buf175 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.int8)
        # Topologically Sorted Source Nodes: [x1_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_18.run(buf174, buf175, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x2_5], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf173, buf28, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 32, 17, 17), (9248, 1, 544, 32))
        buf177 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.int8)
        # Topologically Sorted Source Nodes: [x2_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_18.run(buf176, buf177, 32768, grid=grid(32768), stream=stream0)
        buf178 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf174, buf176, buf178, 65536, grid=grid(65536), stream=stream0)
        buf179 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf178, buf179, 128, 256, grid=grid(128, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, buf29, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf181, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, buf30, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf183 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf182, primals_80, primals_81, primals_82, primals_83, buf183, 32768, grid=grid(32768), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, buf31, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf185 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf185, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, buf32, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf187 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_23.run(buf178, buf187, 128, 256, grid=grid(128, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, buf33, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf189, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, buf34, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf191 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf190, primals_92, primals_93, primals_94, primals_95, buf191, 32768, grid=grid(32768), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, buf35, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf193, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, buf36, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf196 = empty_strided_cuda((4, 32, 2, 16, 16), (16384, 512, 256, 16, 1), torch.float32)
        buf423 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_6, add_3, out_7, x_10], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_24.run(buf186, primals_86, primals_87, primals_88, primals_89, buf194, primals_98, primals_99, primals_100, primals_101, buf178, buf196, buf423, 256, 256, grid=grid(256, 256), stream=stream0)
        buf197 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf196, buf197, 128, 256, grid=grid(128, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, buf37, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf199, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, buf38, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf201 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf200, primals_104, primals_105, primals_106, primals_107, buf201, 32768, grid=grid(32768), stream=stream0)
        del primals_107
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, buf39, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf203 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf203, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, buf40, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf205 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_23.run(buf196, buf205, 128, 256, grid=grid(128, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, buf41, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf207, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, buf42, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf208, primals_116, primals_117, primals_118, primals_119, buf209, 32768, grid=grid(32768), stream=stream0)
        del primals_119
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, buf43, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_21.run(buf211, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, buf44, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf214 = empty_strided_cuda((4, 32, 2, 16, 16), (16384, 512, 256, 16, 1), torch.float32)
        buf422 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.bool)
        # Topologically Sorted Source Nodes: [out_8, add_4, out_9, x_13], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_24.run(buf204, primals_110, primals_111, primals_112, primals_113, buf212, primals_122, primals_123, primals_124, primals_125, buf196, buf214, buf422, 256, 256, grid=grid(256, 256), stream=stream0)
        buf215 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_25.run(buf214, buf215, 256, 256, grid=grid(256, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [x1_9], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, buf45, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 64, 9, 9), (5184, 1, 576, 64))
        buf217 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x1_10], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_26.run(buf216, buf217, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x2_9], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf215, buf46, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 64, 9, 9), (5184, 1, 576, 64))
        buf219 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x2_10], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_26.run(buf218, buf219, 16384, grid=grid(16384), stream=stream0)
        buf220 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf216, buf218, buf220, 32768, grid=grid(32768), stream=stream0)
        buf221 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf220, buf221, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, buf47, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf223, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf48, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf225 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [input_107, input_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf224, primals_130, primals_131, primals_132, primals_133, buf225, 16384, grid=grid(16384), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, buf49, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf227 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf227, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, buf50, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf229 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf220, buf229, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, buf51, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf231, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, buf52, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf233 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [input_117, input_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf232, primals_142, primals_143, primals_144, primals_145, buf233, 16384, grid=grid(16384), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf53, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf235 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf235, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, buf54, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf238 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf421 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_10, add_5, out_11, x_16], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf228, primals_136, primals_137, primals_138, primals_139, buf236, primals_148, primals_149, primals_150, primals_151, buf220, buf238, buf421, 512, 64, grid=grid(512, 64), stream=stream0)
        buf239 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf238, buf239, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, buf55, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf241, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, buf56, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf243 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [input_127, input_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf242, primals_154, primals_155, primals_156, primals_157, buf243, 16384, grid=grid(16384), stream=stream0)
        del primals_157
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, buf57, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf245, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, buf58, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf247 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf238, buf247, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, buf59, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf249, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, buf60, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf251 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [input_137, input_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf250, primals_166, primals_167, primals_168, primals_169, buf251, 16384, grid=grid(16384), stream=stream0)
        del primals_169
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, buf61, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf253, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, buf62, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf256 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf420 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_12, add_6, out_13, x_19], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf246, primals_160, primals_161, primals_162, primals_163, buf254, primals_172, primals_173, primals_174, primals_175, buf238, buf256, buf420, 512, 64, grid=grid(512, 64), stream=stream0)
        buf257 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf256, buf257, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, buf63, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf259 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf259, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_146], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, buf64, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf261 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [input_147, input_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf260, primals_178, primals_179, primals_180, primals_181, buf261, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, buf65, stride=(1, 1), padding=(5, 0), dilation=(5, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf263, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, buf66, stride=(1, 1), padding=(0, 5), dilation=(1, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf265 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf256, buf265, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, buf67, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [input_155], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf267, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, buf68, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf269 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [input_157, input_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf268, primals_190, primals_191, primals_192, primals_193, buf269, 16384, grid=grid(16384), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, buf69, stride=(1, 1), padding=(0, 5), dilation=(1, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf271, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, buf70, stride=(1, 1), padding=(5, 0), dilation=(5, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf274 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf419 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_14, add_7, out_15, x_22], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf264, primals_184, primals_185, primals_186, primals_187, buf272, primals_196, primals_197, primals_198, primals_199, buf256, buf274, buf419, 512, 64, grid=grid(512, 64), stream=stream0)
        buf275 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf274, buf275, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, buf71, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf277, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, buf72, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf279 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [input_167, input_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf278, primals_202, primals_203, primals_204, primals_205, buf279, 16384, grid=grid(16384), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, buf73, stride=(1, 1), padding=(9, 0), dilation=(9, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf281, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, buf74, stride=(1, 1), padding=(0, 9), dilation=(1, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf283 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf274, buf283, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, buf75, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf285 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf285, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, buf76, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf287 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [input_177, input_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf286, primals_214, primals_215, primals_216, primals_217, buf287, 16384, grid=grid(16384), stream=stream0)
        del primals_217
        # Topologically Sorted Source Nodes: [input_179], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, buf77, stride=(1, 1), padding=(0, 9), dilation=(1, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf289, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, buf78, stride=(1, 1), padding=(9, 0), dilation=(9, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf292 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf418 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_16, add_8, out_17, x_25], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf282, primals_208, primals_209, primals_210, primals_211, buf290, primals_220, primals_221, primals_222, primals_223, buf274, buf292, buf418, 512, 64, grid=grid(512, 64), stream=stream0)
        buf293 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf292, buf293, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, buf79, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf295, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, buf80, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf297 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [input_187, input_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf296, primals_226, primals_227, primals_228, primals_229, buf297, 16384, grid=grid(16384), stream=stream0)
        del primals_229
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, buf81, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [input_190], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf299, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, buf82, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf301 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf292, buf301, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, buf83, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf303 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [input_195], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf303, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_196], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, buf84, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf305 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf304, primals_238, primals_239, primals_240, primals_241, buf305, 16384, grid=grid(16384), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [input_199], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, buf85, stride=(1, 1), padding=(0, 2), dilation=(1, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf307 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf307, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, buf86, stride=(1, 1), padding=(2, 0), dilation=(2, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf310 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf417 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_18, add_9, out_19, x_28], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf300, primals_232, primals_233, primals_234, primals_235, buf308, primals_244, primals_245, primals_246, primals_247, buf292, buf310, buf417, 512, 64, grid=grid(512, 64), stream=stream0)
        buf311 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf310, buf311, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, buf87, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf313 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf313, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, buf88, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf315 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [input_207, input_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf314, primals_250, primals_251, primals_252, primals_253, buf315, 16384, grid=grid(16384), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, buf89, stride=(1, 1), padding=(5, 0), dilation=(5, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf317 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf317, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_211], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, buf90, stride=(1, 1), padding=(0, 5), dilation=(1, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf319 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf310, buf319, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf91, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [input_215], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf321, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, buf92, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf323 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [input_217, input_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf322, primals_262, primals_263, primals_264, primals_265, buf323, 16384, grid=grid(16384), stream=stream0)
        del primals_265
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, buf93, stride=(1, 1), padding=(0, 5), dilation=(1, 5), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf325, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_221], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, buf94, stride=(1, 1), padding=(5, 0), dilation=(5, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf328 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf416 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_20, add_10, out_21, x_31], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf318, primals_256, primals_257, primals_258, primals_259, buf326, primals_268, primals_269, primals_270, primals_271, buf310, buf328, buf416, 512, 64, grid=grid(512, 64), stream=stream0)
        buf329 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf328, buf329, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, buf95, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf331 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf331, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, buf96, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf333 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [input_227, input_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf332, primals_274, primals_275, primals_276, primals_277, buf333, 16384, grid=grid(16384), stream=stream0)
        del primals_277
        # Topologically Sorted Source Nodes: [input_229], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, buf97, stride=(1, 1), padding=(9, 0), dilation=(9, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf335, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_231], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, buf98, stride=(1, 1), padding=(0, 9), dilation=(1, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf337 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf328, buf337, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, buf99, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf339 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [input_235], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf339, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf100, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf341 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [input_237, input_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf340, primals_286, primals_287, primals_288, primals_289, buf341, 16384, grid=grid(16384), stream=stream0)
        del primals_289
        # Topologically Sorted Source Nodes: [input_239], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf101, stride=(1, 1), padding=(0, 9), dilation=(1, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf343, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, buf102, stride=(1, 1), padding=(9, 0), dilation=(9, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf346 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf415 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_22, add_11, out_23, x_34], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf336, primals_280, primals_281, primals_282, primals_283, buf344, primals_292, primals_293, primals_294, primals_295, buf328, buf346, buf415, 512, 64, grid=grid(512, 64), stream=stream0)
        buf347 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf346, buf347, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, buf103, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf349, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, buf104, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf351 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [input_247, input_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf350, primals_298, primals_299, primals_300, primals_301, buf351, 16384, grid=grid(16384), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, buf105, stride=(1, 1), padding=(17, 0), dilation=(17, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf353 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [input_250], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf353, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, buf106, stride=(1, 1), padding=(0, 17), dilation=(1, 17), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf355 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf346, buf355, 256, 64, grid=grid(256, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, buf107, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf357 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf357, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_256], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, buf108, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf359 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [input_257, input_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf358, primals_310, primals_311, primals_312, primals_313, buf359, 16384, grid=grid(16384), stream=stream0)
        del primals_313
        # Topologically Sorted Source Nodes: [input_259], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, buf109, stride=(1, 1), padding=(0, 17), dilation=(1, 17), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf361, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, buf110, stride=(1, 1), padding=(17, 0), dilation=(17, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf364 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf414 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.bool)
        # Topologically Sorted Source Nodes: [out_24, add_12, out_25, x_37], Original ATen: [aten.cat, aten.add, aten.relu, aten.clone, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_relu_threshold_backward_32.run(buf354, primals_304, primals_305, primals_306, primals_307, buf362, primals_316, primals_317, primals_318, primals_319, buf346, buf364, buf414, 512, 64, grid=grid(512, 64), stream=stream0)
        buf365 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf401 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf402 = reinterpret_tensor(buf401, (4, 128, 1, 1), (128, 1, 128, 128), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [x_37, x_38, input_264], Original ATen: [aten.clone, aten.view, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_mean_view_33.run(buf402, buf364, buf365, 512, 64, grid=grid(512), stream=stream0)
        del buf364
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, buf111, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf367 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf366, primals_321, primals_322, primals_323, primals_324, buf367, 8192, grid=grid(8192), stream=stream0)
        del primals_324
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, buf112, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (4, 128, 2, 2), (512, 1, 256, 128))
        buf369 = empty_strided_cuda((4, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf368, primals_326, primals_327, primals_328, primals_329, buf369, 2048, grid=grid(2048), stream=stream0)
        del primals_329
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, buf113, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 128, 1, 1), (128, 1, 128, 128))
        buf371 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf370, primals_331, primals_332, primals_333, primals_334, buf371, 512, grid=grid(512), stream=stream0)
        del primals_334
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 4, 1, 1), (4, 1, 4, 4))
        buf373 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf372, primals_336, primals_337, primals_338, primals_339, buf373, 16, grid=grid(16), stream=stream0)
        buf374 = empty_strided_cuda((2, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_38.run(buf374, 2, grid=grid(2), stream=stream0)
        buf375 = empty_strided_cuda((2, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_38.run(buf375, 2, grid=grid(2), stream=stream0)
        buf376 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_38.run(buf376, 2, grid=grid(2), stream=stream0)
        buf377 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_38.run(buf377, 2, grid=grid(2), stream=stream0)
        buf378 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_39.run(buf378, 2, grid=grid(2), stream=stream0)
        buf380 = empty_strided_cuda((2, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_39.run(buf380, 2, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf369, primals_340, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 4, 2, 2), (16, 1, 8, 4))
        buf379 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf382 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [out_26, x_52, x_53, out_27], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_40.run(buf382, buf374, buf376, buf373, buf377, buf378, buf381, primals_341, primals_342, primals_343, primals_344, buf375, buf380, 16, 4, grid=grid(16, 4), stream=stream0)
        buf383 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(buf383, 4, grid=grid(4), stream=stream0)
        buf384 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_42.run(buf384, 4, grid=grid(4), stream=stream0)
        buf385 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(buf385, 4, grid=grid(4), stream=stream0)
        buf386 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_42.run(buf386, 4, grid=grid(4), stream=stream0)
        buf387 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_43.run(buf387, 4, grid=grid(4), stream=stream0)
        buf389 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_43.run(buf389, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf367, primals_345, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 4, 4, 4), (64, 1, 16, 4))
        buf388 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf391 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [out_28, x_55, x_56, out_29], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44.run(buf391, buf383, buf385, buf382, buf386, buf387, buf390, primals_346, primals_347, primals_348, primals_349, buf384, buf389, 16, 16, grid=grid(16, 16), stream=stream0)
        buf392 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf392, 8, grid=grid(8), stream=stream0)
        buf393 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_46.run(buf393, 8, grid=grid(8), stream=stream0)
        buf394 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf394, 8, grid=grid(8), stream=stream0)
        buf395 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_46.run(buf395, 8, grid=grid(8), stream=stream0)
        buf396 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_47.run(buf396, 8, grid=grid(8), stream=stream0)
        buf398 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_47.run(buf398, 8, grid=grid(8), stream=stream0)
        buf399 = empty_strided_cuda((4, 4, 8, 8), (256, 1, 32, 4), torch.float32)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_48.run(buf392, buf394, buf391, buf395, buf396, buf393, buf398, buf399, 16, 64, grid=grid(16, 64), stream=stream0)
        del buf391
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf365, primals_350, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 4, 8, 8), (256, 1, 32, 4))
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 4, 1, 1), (4, 1, 4, 4))
        buf404 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf403, primals_356, primals_357, primals_358, primals_359, buf404, 16, grid=grid(16), stream=stream0)
        buf405 = empty_strided_cuda((4, 4, 8, 8), (256, 1, 32, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_58, x_59, out_31, x_61, x_62, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_49.run(buf404, buf400, primals_351, primals_352, primals_353, primals_354, buf399, buf405, 1024, grid=grid(1024), stream=stream0)
        del buf404
        buf406 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_50.run(buf406, 64, grid=grid(64), stream=stream0)
        buf407 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_51.run(buf407, 64, grid=grid(64), stream=stream0)
        buf408 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_50.run(buf408, 64, grid=grid(64), stream=stream0)
        buf409 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_51.run(buf409, 64, grid=grid(64), stream=stream0)
        buf410 = reinterpret_tensor(buf382, (64, ), (1, ), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_52.run(buf410, 64, grid=grid(64), stream=stream0)
        buf412 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_52.run(buf412, 64, grid=grid(64), stream=stream0)
        buf411 = reinterpret_tensor(buf214, (4, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf214  # reuse
        buf413 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_53.run(buf413, buf406, buf408, buf405, buf409, buf410, buf407, buf412, 65536, grid=grid(65536), stream=stream0)
        del buf405
    return (buf413, buf0, buf1, buf2, buf3, buf4, primals_6, primals_7, primals_8, buf5, buf6, primals_12, primals_13, primals_14, primals_15, buf7, buf8, primals_18, primals_19, primals_20, buf9, buf10, primals_24, primals_25, primals_26, primals_27, buf11, buf12, primals_30, primals_31, primals_32, buf13, buf14, primals_36, primals_37, primals_38, primals_39, buf15, buf16, primals_42, primals_43, primals_44, buf17, buf18, primals_48, primals_49, primals_50, primals_51, buf19, buf20, primals_54, primals_55, primals_56, buf21, buf22, primals_60, primals_61, primals_62, primals_63, buf23, buf24, primals_66, primals_67, primals_68, buf25, buf26, primals_72, primals_73, primals_74, primals_75, buf27, buf28, buf29, buf30, primals_80, primals_81, primals_82, buf31, buf32, primals_86, primals_87, primals_88, primals_89, buf33, buf34, primals_92, primals_93, primals_94, buf35, buf36, primals_98, primals_99, primals_100, primals_101, buf37, buf38, primals_104, primals_105, primals_106, buf39, buf40, primals_110, primals_111, primals_112, primals_113, buf41, buf42, primals_116, primals_117, primals_118, buf43, buf44, primals_122, primals_123, primals_124, primals_125, buf45, buf46, buf47, buf48, primals_130, primals_131, primals_132, buf49, buf50, primals_136, primals_137, primals_138, primals_139, buf51, buf52, primals_142, primals_143, primals_144, buf53, buf54, primals_148, primals_149, primals_150, primals_151, buf55, buf56, primals_154, primals_155, primals_156, buf57, buf58, primals_160, primals_161, primals_162, primals_163, buf59, buf60, primals_166, primals_167, primals_168, buf61, buf62, primals_172, primals_173, primals_174, primals_175, buf63, buf64, primals_178, primals_179, primals_180, buf65, buf66, primals_184, primals_185, primals_186, primals_187, buf67, buf68, primals_190, primals_191, primals_192, buf69, buf70, primals_196, primals_197, primals_198, primals_199, buf71, buf72, primals_202, primals_203, primals_204, buf73, buf74, primals_208, primals_209, primals_210, primals_211, buf75, buf76, primals_214, primals_215, primals_216, buf77, buf78, primals_220, primals_221, primals_222, primals_223, buf79, buf80, primals_226, primals_227, primals_228, buf81, buf82, primals_232, primals_233, primals_234, primals_235, buf83, buf84, primals_238, primals_239, primals_240, buf85, buf86, primals_244, primals_245, primals_246, primals_247, buf87, buf88, primals_250, primals_251, primals_252, buf89, buf90, primals_256, primals_257, primals_258, primals_259, buf91, buf92, primals_262, primals_263, primals_264, buf93, buf94, primals_268, primals_269, primals_270, primals_271, buf95, buf96, primals_274, primals_275, primals_276, buf97, buf98, primals_280, primals_281, primals_282, primals_283, buf99, buf100, primals_286, primals_287, primals_288, buf101, buf102, primals_292, primals_293, primals_294, primals_295, buf103, buf104, primals_298, primals_299, primals_300, buf105, buf106, primals_304, primals_305, primals_306, primals_307, buf107, buf108, primals_310, primals_311, primals_312, buf109, buf110, primals_316, primals_317, primals_318, primals_319, buf111, primals_321, primals_322, primals_323, buf112, primals_326, primals_327, primals_328, buf113, primals_331, primals_332, primals_333, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, buf114, buf115, buf116, buf117, reinterpret_tensor(buf118, (4, 16, 32, 32), (32768, 1024, 32, 1), 0), reinterpret_tensor(buf118, (4, 16, 32, 32), (32768, 1024, 32, 1), 16384), buf121, buf122, buf123, buf125, buf126, buf129, buf130, buf131, buf133, buf134, reinterpret_tensor(buf136, (4, 16, 32, 32), (32768, 1024, 32, 1), 0), reinterpret_tensor(buf136, (4, 16, 32, 32), (32768, 1024, 32, 1), 16384), buf139, buf140, buf141, buf143, buf144, buf147, buf148, buf149, buf151, buf152, reinterpret_tensor(buf154, (4, 16, 32, 32), (32768, 1024, 32, 1), 0), reinterpret_tensor(buf154, (4, 16, 32, 32), (32768, 1024, 32, 1), 16384), buf157, buf158, buf159, buf161, buf162, buf165, buf166, buf167, buf169, buf170, buf173, buf174, buf175, buf176, buf177, reinterpret_tensor(buf178, (4, 32, 16, 16), (16384, 256, 16, 1), 0), reinterpret_tensor(buf178, (4, 32, 16, 16), (16384, 256, 16, 1), 8192), buf181, buf182, buf183, buf185, buf186, buf189, buf190, buf191, buf193, buf194, reinterpret_tensor(buf196, (4, 32, 16, 16), (16384, 256, 16, 1), 0), reinterpret_tensor(buf196, (4, 32, 16, 16), (16384, 256, 16, 1), 8192), buf199, buf200, buf201, buf203, buf204, buf207, buf208, buf209, buf211, buf212, buf215, buf216, buf217, buf218, buf219, reinterpret_tensor(buf220, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf220, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf223, buf224, buf225, buf227, buf228, buf231, buf232, buf233, buf235, buf236, reinterpret_tensor(buf238, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf238, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf241, buf242, buf243, buf245, buf246, buf249, buf250, buf251, buf253, buf254, reinterpret_tensor(buf256, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf256, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf259, buf260, buf261, buf263, buf264, buf267, buf268, buf269, buf271, buf272, reinterpret_tensor(buf274, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf274, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf277, buf278, buf279, buf281, buf282, buf285, buf286, buf287, buf289, buf290, reinterpret_tensor(buf292, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf292, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf295, buf296, buf297, buf299, buf300, buf303, buf304, buf305, buf307, buf308, reinterpret_tensor(buf310, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf310, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf313, buf314, buf315, buf317, buf318, buf321, buf322, buf323, buf325, buf326, reinterpret_tensor(buf328, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf328, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf331, buf332, buf333, buf335, buf336, buf339, buf340, buf341, buf343, buf344, reinterpret_tensor(buf346, (4, 64, 8, 8), (8192, 64, 8, 1), 0), reinterpret_tensor(buf346, (4, 64, 8, 8), (8192, 64, 8, 1), 4096), buf349, buf350, buf351, buf353, buf354, buf357, buf358, buf359, buf361, buf362, buf365, buf366, buf367, buf368, buf369, buf370, buf371, buf372, buf374, buf375, buf376, buf377, buf378, buf380, buf381, buf383, buf384, buf385, buf386, buf387, buf389, buf390, buf392, buf393, buf394, buf395, buf396, buf398, buf399, buf400, buf402, buf403, buf406, buf407, buf408, buf409, buf410, buf412, buf414, buf415, buf416, buf417, buf418, buf419, buf420, buf421, buf422, buf423, buf424, buf425, buf426, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((16, 16, 1, 3), (48, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((16, 16, 3, 1), (48, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, 32, 1, 3), (96, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, 32, 3, 1), (96, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, 64, 1, 3), (192, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, 64, 3, 1), (192, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, 128, 7, 7), (6272, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
