# AOT ID: ['7_forward']
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


# kernel path: inductor_cache/vl/cvluqfigqu7hr34s4j3x5s6h2cgckevd74d3xjlnratfe63esdkw.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/p6/cp6upnlpsmgeslmbj7yh4hd5k7qmbchg3e73dpwyye7npjlpn6hm.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
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


# kernel path: inductor_cache/va/cva2uq2facb2dqsnhpesm4azv7g7qnyxeysourvddza6rnwdzejy.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xp/cxp6xctk6srjx2ysm5zjegyicaf7iltamuhbfrknrx7wf224hfkl.py
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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/lj/cljpx2ip2pik2mfddi2ggcv7cilpr7lffpjnj2dpk442ytehwrbd.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/g3/cg3kc6tn6eqyqwnp5ibo56hlartk35hvmfv6mgfkzgam62jy6daw.py
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
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 3456*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5iydelol7c3ecogzdkdqlj3obervbx5mo2xn246ebe45ly4f2df.py
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
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 589824
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 768*x2 + 6912*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crfwpwzircmp5grbh66apzbzbucr63ila3hjqrybtpp2remthr7k.py
# Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
triton_poi_fused_div_sub_8 = async_compile.triton('triton_poi_fused_div_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mk/cmkfyb5efiotwj2v3fbw4zlu6igcpqyizi4osyg3xd7poctbzuay.py
# Topologically Sorted Source Nodes: [batch_norm, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => mul_1, sub_1
#   out_1 => relu
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/zl/czlly5ynyebb55r3ahliylsimrtu27clp72pmbbv454bqlnxi5te.py
# Topologically Sorted Source Nodes: [out_4, batch_norm_2, out_5], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => mul_5, sub_3
#   out_4 => add_2
#   out_5 => relu_2
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %relu), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_11), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
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
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyyftk45tj773h7zjt3qdq5lyywyynrmwfmk7vb3fvwfqqseeutd.py
# Topologically Sorted Source Nodes: [batch_norm_6, relu_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_6 => mul_13, sub_7
#   relu_5 => relu_5
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_27), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mul_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/ri/cridh4gwnzyd5bpsehvw3y7paouy2ezkjsf5nn5vn5q3gvzhglux.py
# Topologically Sorted Source Nodes: [input_2, out_12, batch_norm_7, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_7 => mul_15, sub_8
#   input_2 => mul_11, sub_6
#   out_12 => add_9
#   out_13 => relu_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_21), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_23), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %mul_11), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %unsqueeze_29), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_31), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp13 = tmp0 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp16 + tmp5
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp8 / tmp18
    tmp20 = tmp19 * tmp10
    tmp21 = tmp15 * tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ot/cotpvxdie4nmqairnektvokbnj6fms3ww4xlmib3ldlctq4gly63.py
# Topologically Sorted Source Nodes: [out_16, batch_norm_9, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_9 => mul_19, sub_10
#   out_16 => add_12
#   out_17 => relu_8
# Graph fragment:
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %relu_6), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %unsqueeze_37), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_39), kwargs = {})
#   %relu_8 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
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
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/y4/cy4tpc3jkdcm25as3mqvjcielgqda5iiinuhoeev3cfbh2v4mhnb.py
# Topologically Sorted Source Nodes: [batch_norm_11, relu_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => mul_23, sub_12
#   relu_9 => relu_9
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_45), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_47), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mul_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/zh/czhkc42xxd6ygmzs3hhza7cgpaufbieoizmahhpdonpj3tv2625i.py
# Topologically Sorted Source Nodes: [input_4, out_20, batch_norm_12, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_12 => mul_25, sub_13
#   input_4 => mul_21, sub_11
#   out_20 => add_16
#   out_21 => relu_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_43), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %mul_21), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %unsqueeze_49), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_51), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_25,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp13 = tmp0 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp16 + tmp5
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp8 / tmp18
    tmp20 = tmp19 * tmp10
    tmp21 = tmp15 * tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/nr/cnrevblhtm2t5ec35mcbbctxclfy7yd2wbebhswout4jxvwdnohb.py
# Topologically Sorted Source Nodes: [out_24, batch_norm_14, out_25], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_14 => mul_29, sub_15
#   out_24 => add_19
#   out_25 => relu_12
# Graph fragment:
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %relu_10), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %unsqueeze_57), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_59), kwargs = {})
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
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
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/af/cafwby7cei2wgabm3s7vfk442b4ftpuah3ikjnpbhcptyjxxzjux.py
# Topologically Sorted Source Nodes: [batch_norm_16, relu_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_16 => mul_33, sub_17
#   relu_13 => relu_13
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_67), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mul_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/zn/cznni2fpnj4ihgr3k42ifn773etcgib3f25bdhydtfkkg4malf52.py
# Topologically Sorted Source Nodes: [input_6, out_28, batch_norm_17, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_17 => mul_35, sub_18
#   input_6 => mul_31, sub_16
#   out_28 => add_23
#   out_29 => relu_14
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_61), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_63), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %mul_31), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_23, %unsqueeze_69), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_71), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%mul_35,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp13 = tmp0 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp16 + tmp5
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp8 / tmp18
    tmp20 = tmp19 * tmp10
    tmp21 = tmp15 * tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/ac/cacwbk7t2r5e3dx2p65ubdorl2a3fqyyjopk4vwxv3yddjkbwixk.py
# Topologically Sorted Source Nodes: [out_32, batch_norm_19, out_33], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_19 => mul_39, sub_20
#   out_32 => add_26
#   out_33 => relu_16
# Graph fragment:
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %relu_14), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_26, %unsqueeze_77), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_79), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mul_39,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
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
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/uv/cuv72jvqnb3sc7zbrcadganmi46u6rb6yktomwh6uquq6wazl6km.py
# Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   out_34 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_16, [4, 4]), kwargs = {})
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_poi_fused_avg_pool2d_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3072*x1 + 24576*x2), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + 3072*x1 + 24576*x2), None)
    tmp3 = tl.load(in_ptr0 + (1536 + x0 + 3072*x1 + 24576*x2), None)
    tmp5 = tl.load(in_ptr0 + (2304 + x0 + 3072*x1 + 24576*x2), None)
    tmp7 = tl.load(in_ptr0 + (6144 + x0 + 3072*x1 + 24576*x2), None)
    tmp9 = tl.load(in_ptr0 + (6912 + x0 + 3072*x1 + 24576*x2), None)
    tmp11 = tl.load(in_ptr0 + (7680 + x0 + 3072*x1 + 24576*x2), None)
    tmp13 = tl.load(in_ptr0 + (8448 + x0 + 3072*x1 + 24576*x2), None)
    tmp15 = tl.load(in_ptr0 + (12288 + x0 + 3072*x1 + 24576*x2), None)
    tmp17 = tl.load(in_ptr0 + (13056 + x0 + 3072*x1 + 24576*x2), None)
    tmp19 = tl.load(in_ptr0 + (13824 + x0 + 3072*x1 + 24576*x2), None)
    tmp21 = tl.load(in_ptr0 + (14592 + x0 + 3072*x1 + 24576*x2), None)
    tmp23 = tl.load(in_ptr0 + (18432 + x0 + 3072*x1 + 24576*x2), None)
    tmp25 = tl.load(in_ptr0 + (19200 + x0 + 3072*x1 + 24576*x2), None)
    tmp27 = tl.load(in_ptr0 + (19968 + x0 + 3072*x1 + 24576*x2), None)
    tmp29 = tl.load(in_ptr0 + (20736 + x0 + 3072*x1 + 24576*x2), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x3), tmp32, None)
''', device_str='cuda')


# kernel path: inductor_cache/se/cseyyb5kgoucejqnrbpnsmnaodupzzscoz6vji5yymzi33jxh6el.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out_37 => amax
# Graph fragment:
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
triton_red_fused__softmax_21 = async_compile.triton('triton_red_fused__softmax_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_21(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x1 = xindex // 4
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (25*x0 + 100*((r2 % 4)) + 400*x1 + (r2 // 4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/he/chehogbvjperd2nmn2e7hqkwmhy7blcko6522hwumw2ppwchrn7c.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out_37 => amax
# Graph fragment:
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
triton_per_fused__softmax_22 = async_compile.triton('triton_per_fused__softmax_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_22(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/he/cheqlbnjpoau73oa6ccbyuyfcpm2erqtyri2jee4zktmtjl6n23c.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out_37 => exp, sub_21, sum_1
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_21,), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__softmax_23 = async_compile.triton('triton_red_fused__softmax_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_23(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (25*x0 + 100*((r2 % 4)) + 400*x1 + (r2 // 4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciecbp6qhmy77juqxvfrnzyxa4jlvcyncbzvfb5iwbsdmzkl2p5r.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out_37 => exp, sub_21, sum_1
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_21,), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_per_fused__softmax_24 = async_compile.triton('triton_per_fused__softmax_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cnsgchsjfkx5b3qhfp7wkbrdpbgdshzhx64wetkb7qg7qz4tzily.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out_37 => div_1, exp, sub_21
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_21,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_25 = async_compile.triton('triton_poi_fused__softmax_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 400)
    x1 = xindex // 400
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (100*((x0 % 4)) + 400*x1 + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cwwis667vqu7f4d3u4v3ohqhybmnoc7bxn2fpeved4vp2u4izog6.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [0]), kwargs = {})
triton_poi_fused_mean_26 = async_compile.triton('triton_poi_fused_mean_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1600 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (3200 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (4800 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (6400 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (8000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = 6.0
    tmp12 = tmp10 / tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_4, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (192, ), (1, ))
    assert_size_stride(primals_25, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (384, ), (1, ))
    assert_size_stride(primals_37, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_44, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_49, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (100, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_65, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (192, ), (1, ))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_83, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_84, (192, ), (1, ))
    assert_size_stride(primals_85, (192, ), (1, ))
    assert_size_stride(primals_86, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_87, (192, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (192, ), (1, ))
    assert_size_stride(primals_92, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_93, (192, ), (1, ))
    assert_size_stride(primals_94, (192, ), (1, ))
    assert_size_stride(primals_95, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_98, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_104, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_108, (384, ), (1, ))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (100, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_126, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, ), (1, ))
    assert_size_stride(primals_135, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_142, (192, ), (1, ))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_148, (192, ), (1, ))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_150, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (100, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_187, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, ), (1, ))
    assert_size_stride(primals_193, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_194, (64, ), (1, ))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_200, (64, ), (1, ))
    assert_size_stride(primals_201, (64, ), (1, ))
    assert_size_stride(primals_202, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (192, ), (1, ))
    assert_size_stride(primals_208, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_209, (192, ), (1, ))
    assert_size_stride(primals_210, (192, ), (1, ))
    assert_size_stride(primals_211, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_212, (192, ), (1, ))
    assert_size_stride(primals_213, (192, ), (1, ))
    assert_size_stride(primals_214, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_215, (192, ), (1, ))
    assert_size_stride(primals_216, (192, ), (1, ))
    assert_size_stride(primals_217, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_218, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_229, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_238, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_240, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_242, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_244, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_245, (768, ), (1, ))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (100, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_248, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (64, ), (1, ))
    assert_size_stride(primals_263, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_264, (192, ), (1, ))
    assert_size_stride(primals_265, (192, ), (1, ))
    assert_size_stride(primals_266, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_267, (192, ), (1, ))
    assert_size_stride(primals_268, (192, ), (1, ))
    assert_size_stride(primals_269, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_270, (192, ), (1, ))
    assert_size_stride(primals_271, (192, ), (1, ))
    assert_size_stride(primals_272, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_276, (192, ), (1, ))
    assert_size_stride(primals_277, (192, ), (1, ))
    assert_size_stride(primals_278, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_282, (384, ), (1, ))
    assert_size_stride(primals_283, (384, ), (1, ))
    assert_size_stride(primals_284, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_288, (384, ), (1, ))
    assert_size_stride(primals_289, (384, ), (1, ))
    assert_size_stride(primals_290, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (384, ), (1, ))
    assert_size_stride(primals_293, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_294, (768, ), (1, ))
    assert_size_stride(primals_295, (768, ), (1, ))
    assert_size_stride(primals_296, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_298, (768, ), (1, ))
    assert_size_stride(primals_299, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_300, (768, ), (1, ))
    assert_size_stride(primals_301, (768, ), (1, ))
    assert_size_stride(primals_302, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (768, ), (1, ))
    assert_size_stride(primals_305, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_306, (768, ), (1, ))
    assert_size_stride(primals_307, (768, ), (1, ))
    assert_size_stride(primals_308, (100, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_309, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (64, ), (1, ))
    assert_size_stride(primals_312, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (64, ), (1, ))
    assert_size_stride(primals_318, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, ), (1, ))
    assert_size_stride(primals_321, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_322, (64, ), (1, ))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_325, (192, ), (1, ))
    assert_size_stride(primals_326, (192, ), (1, ))
    assert_size_stride(primals_327, (192, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_328, (192, ), (1, ))
    assert_size_stride(primals_329, (192, ), (1, ))
    assert_size_stride(primals_330, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_331, (192, ), (1, ))
    assert_size_stride(primals_332, (192, ), (1, ))
    assert_size_stride(primals_333, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_334, (192, ), (1, ))
    assert_size_stride(primals_335, (192, ), (1, ))
    assert_size_stride(primals_336, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_337, (192, ), (1, ))
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_340, (384, ), (1, ))
    assert_size_stride(primals_341, (384, ), (1, ))
    assert_size_stride(primals_342, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_343, (384, ), (1, ))
    assert_size_stride(primals_344, (384, ), (1, ))
    assert_size_stride(primals_345, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_346, (384, ), (1, ))
    assert_size_stride(primals_347, (384, ), (1, ))
    assert_size_stride(primals_348, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_349, (384, ), (1, ))
    assert_size_stride(primals_350, (384, ), (1, ))
    assert_size_stride(primals_351, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_352, (384, ), (1, ))
    assert_size_stride(primals_353, (384, ), (1, ))
    assert_size_stride(primals_354, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_355, (768, ), (1, ))
    assert_size_stride(primals_356, (768, ), (1, ))
    assert_size_stride(primals_357, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_358, (768, ), (1, ))
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_361, (768, ), (1, ))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_364, (768, ), (1, ))
    assert_size_stride(primals_365, (768, ), (1, ))
    assert_size_stride(primals_366, (768, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (100, 768, 1, 1), (768, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_4, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_7, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_10, buf2, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_10
        buf3 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_13, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_13
        buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_16, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_16
        buf5 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_22, buf5, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_22
        buf6 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_25, buf6, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_25
        buf7 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_28, buf7, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_28
        buf8 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_31, buf8, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_31
        buf9 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_37, buf9, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_37
        buf10 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_40, buf10, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_40
        buf11 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_43, buf11, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_43
        buf12 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_46, buf12, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_46
        buf13 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_52, buf13, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_52
        buf14 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_55, buf14, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_55
        buf15 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_58, buf15, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_58
        buf16 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_61, buf16, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_61
        buf17 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_65, buf17, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_65
        buf18 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_68, buf18, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_68
        buf19 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_71, buf19, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_71
        buf20 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_74, buf20, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_74
        buf21 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_77, buf21, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_77
        buf22 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_83, buf22, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_83
        buf23 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_86, buf23, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_86
        buf24 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_89, buf24, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_89
        buf25 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_92, buf25, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_92
        buf26 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_98, buf26, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_98
        buf27 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_101, buf27, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_101
        buf28 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_104, buf28, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_104
        buf29 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_107, buf29, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_107
        buf30 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_113, buf30, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_113
        buf31 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_116, buf31, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_116
        buf32 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_119, buf32, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_119
        buf33 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_122, buf33, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_122
        buf34 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_126, buf34, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_126
        buf35 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_129, buf35, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_129
        buf36 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_132, buf36, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_132
        buf37 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_135, buf37, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_135
        buf38 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_138, buf38, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_138
        buf39 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_144, buf39, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_144
        buf40 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_147, buf40, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_147
        buf41 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_150, buf41, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_150
        buf42 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_153, buf42, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_153
        buf43 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_159, buf43, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_159
        buf44 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_162, buf44, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_162
        buf45 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_165, buf45, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_165
        buf46 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_168, buf46, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_168
        buf47 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_174, buf47, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_174
        buf48 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_177, buf48, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_177
        buf49 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_180, buf49, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_180
        buf50 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_183, buf50, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_183
        buf51 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_187, buf51, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_187
        buf52 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_190, buf52, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_190
        buf53 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_193, buf53, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_193
        buf54 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_196, buf54, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_196
        buf55 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_199, buf55, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_199
        buf56 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_205, buf56, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_205
        buf57 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_208, buf57, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_208
        buf58 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_211, buf58, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_211
        buf59 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_214, buf59, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_214
        buf60 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_220, buf60, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_220
        buf61 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_223, buf61, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_223
        buf62 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_226, buf62, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_226
        buf63 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_229, buf63, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_229
        buf64 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_235, buf64, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_235
        buf65 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_238, buf65, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_238
        buf66 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_241, buf66, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_241
        buf67 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_244, buf67, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_244
        buf68 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_248, buf68, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_248
        buf69 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_251, buf69, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_251
        buf70 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_254, buf70, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_254
        buf71 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_257, buf71, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_257
        buf72 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_260, buf72, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_260
        buf73 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_266, buf73, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_266
        buf74 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_269, buf74, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_269
        buf75 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_272, buf75, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_272
        buf76 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_275, buf76, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_275
        buf77 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_281, buf77, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_281
        buf78 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_284, buf78, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_284
        buf79 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_287, buf79, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_287
        buf80 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_290, buf80, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_290
        buf81 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_296, buf81, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_296
        buf82 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_299, buf82, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_299
        buf83 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_302, buf83, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_302
        buf84 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_305, buf84, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_305
        buf85 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_309, buf85, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_309
        buf86 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_312, buf86, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_312
        buf87 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_315, buf87, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_315
        buf88 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_318, buf88, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_318
        buf89 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_321, buf89, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_321
        buf90 = empty_strided_cuda((192, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_327, buf90, 12288, 9, grid=grid(12288, 9), stream=stream0)
        del primals_327
        buf91 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_330, buf91, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_330
        buf92 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_333, buf92, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_333
        buf93 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_336, buf93, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_336
        buf94 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_342, buf94, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_342
        buf95 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_345, buf95, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_345
        buf96 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_348, buf96, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_348
        buf97 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_351, buf97, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del primals_351
        buf98 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_357, buf98, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del primals_357
        buf99 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_360, buf99, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_360
        buf100 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_363, buf100, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_363
        buf101 = empty_strided_cuda((768, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_366, buf101, 589824, 9, grid=grid(589824, 9), stream=stream0)
        del primals_366
        buf102 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_8.run(primals_2, primals_1, primals_3, buf102, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 64, 64), (262144, 1, 4096, 64))
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf102, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 64, 64, 64), (262144, 1, 4096, 64))
        # Topologically Sorted Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf102, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 64, 64, 64), (262144, 1, 4096, 64))
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf102, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 64, 64, 64), (262144, 1, 4096, 64))
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf102, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 64, 64, 64), (262144, 1, 4096, 64))
        # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf102, buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [batch_norm, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf104, primals_5, primals_6, 1048576, grid=grid(1048576), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_20, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf147, primals_66, primals_67, 1048576, grid=grid(1048576), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_40, out_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf190, primals_127, primals_128, 1048576, grid=grid(1048576), stream=stream0)
        del primals_127
        # Topologically Sorted Source Nodes: [out_78], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_60, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf233, primals_188, primals_189, 1048576, grid=grid(1048576), stream=stream0)
        del primals_188
        # Topologically Sorted Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_80, out_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf276, primals_249, primals_250, 1048576, grid=grid(1048576), stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [out_154], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf319 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_100, out_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf319, primals_310, primals_311, 1048576, grid=grid(1048576), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [out_192], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf106, primals_8, primals_9, 1048576, grid=grid(1048576), stream=stream0)
        del primals_8
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf108 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [out_4, batch_norm_2, out_5], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf108, buf104, primals_11, primals_12, 1048576, grid=grid(1048576), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_21, relu_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf149, primals_69, primals_70, 1048576, grid=grid(1048576), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_42, batch_norm_22, out_43], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf151, buf147, primals_72, primals_73, 1048576, grid=grid(1048576), stream=stream0)
        del primals_72
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_41, relu_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf192, primals_130, primals_131, 1048576, grid=grid(1048576), stream=stream0)
        del primals_130
        # Topologically Sorted Source Nodes: [out_79], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf194 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [out_80, batch_norm_42, out_81], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf194, buf190, primals_133, primals_134, 1048576, grid=grid(1048576), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [out_82], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf235 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_61, relu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf235, primals_191, primals_192, 1048576, grid=grid(1048576), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [out_118, batch_norm_62, out_119], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf237, buf233, primals_194, primals_195, 1048576, grid=grid(1048576), stream=stream0)
        del primals_194
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf278 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_81, relu_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf278, primals_252, primals_253, 1048576, grid=grid(1048576), stream=stream0)
        del primals_252
        # Topologically Sorted Source Nodes: [out_155], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [out_156, batch_norm_82, out_157], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf280, buf276, primals_255, primals_256, 1048576, grid=grid(1048576), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [out_158], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_101, relu_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf321, primals_313, primals_314, 1048576, grid=grid(1048576), stream=stream0)
        del primals_313
        # Topologically Sorted Source Nodes: [out_193], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [out_194, batch_norm_102, out_195], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf323, buf319, primals_316, primals_317, 1048576, grid=grid(1048576), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf110, primals_14, primals_15, 1048576, grid=grid(1048576), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [out_8, batch_norm_4, out_9], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf112, buf108, primals_17, primals_18, 1048576, grid=grid(1048576), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf112, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_23, relu_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf153, primals_75, primals_76, 1048576, grid=grid(1048576), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [out_46, batch_norm_24, out_47], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf155, buf151, primals_78, primals_79, 1048576, grid=grid(1048576), stream=stream0)
        del primals_78
        # Topologically Sorted Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf155, buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_43, relu_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf196, primals_136, primals_137, 1048576, grid=grid(1048576), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [out_84, batch_norm_44, out_85], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf198, buf194, primals_139, primals_140, 1048576, grid=grid(1048576), stream=stream0)
        del primals_139
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf198, buf39, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_63, relu_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf239, primals_197, primals_198, 1048576, grid=grid(1048576), stream=stream0)
        del primals_197
        # Topologically Sorted Source Nodes: [out_121], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [out_122, batch_norm_64, out_123], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf241, buf237, primals_200, primals_201, 1048576, grid=grid(1048576), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf241, buf56, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_83, relu_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf282, primals_258, primals_259, 1048576, grid=grid(1048576), stream=stream0)
        del primals_258
        # Topologically Sorted Source Nodes: [out_159], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [out_160, batch_norm_84, out_161], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf284, buf280, primals_261, primals_262, 1048576, grid=grid(1048576), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [out_162], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf73, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_103, relu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf325, primals_319, primals_320, 1048576, grid=grid(1048576), stream=stream0)
        del primals_319
        # Topologically Sorted Source Nodes: [out_197], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 64, 64, 64), (262144, 1, 4096, 64))
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [out_198, batch_norm_104, out_199], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf327, buf323, primals_322, primals_323, 1048576, grid=grid(1048576), stream=stream0)
        del primals_322
        # Topologically Sorted Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf327, buf90, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_19, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_6, relu_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf115, primals_23, primals_24, 786432, grid=grid(786432), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_80, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_26, relu_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf158, primals_84, primals_85, 786432, grid=grid(786432), stream=stream0)
        del primals_84
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_141, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_46, relu_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf201, primals_145, primals_146, 786432, grid=grid(786432), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [out_87], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_202, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_66, relu_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf244, primals_206, primals_207, 786432, grid=grid(786432), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_125], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_263, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_86, relu_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf287, primals_267, primals_268, 786432, grid=grid(786432), stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 192, 32, 32), (196608, 1, 6144, 192))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_324, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_106, relu_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf330, primals_328, primals_329, 786432, grid=grid(786432), stream=stream0)
        del primals_328
        # Topologically Sorted Source Nodes: [out_201], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [input_2, out_12, batch_norm_7, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf117, buf113, primals_20, primals_21, primals_26, primals_27, 786432, grid=grid(786432), stream=stream0)
        del buf113
        del primals_20
        del primals_26
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [input_8, out_50, batch_norm_27, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf160, buf156, primals_81, primals_82, primals_87, primals_88, 786432, grid=grid(786432), stream=stream0)
        del buf156
        del primals_81
        del primals_87
        # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf203 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [input_14, out_88, batch_norm_47, out_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf203, buf199, primals_142, primals_143, primals_148, primals_149, 786432, grid=grid(786432), stream=stream0)
        del buf199
        del primals_142
        del primals_148
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [input_20, out_126, batch_norm_67, out_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf246, buf242, primals_203, primals_204, primals_209, primals_210, 786432, grid=grid(786432), stream=stream0)
        del buf242
        del primals_203
        del primals_209
        # Topologically Sorted Source Nodes: [out_128], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [input_26, out_164, batch_norm_87, out_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf289, buf285, primals_264, primals_265, primals_270, primals_271, 786432, grid=grid(786432), stream=stream0)
        del buf285
        del primals_264
        del primals_270
        # Topologically Sorted Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [input_32, out_202, batch_norm_107, out_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf332, buf328, primals_325, primals_326, primals_331, primals_332, 786432, grid=grid(786432), stream=stream0)
        del buf328
        del primals_325
        del primals_331
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, buf92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_8, relu_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf119, primals_29, primals_30, 786432, grid=grid(786432), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [out_16, batch_norm_9, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf121, buf117, primals_32, primals_33, 786432, grid=grid(786432), stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_28, relu_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf162, primals_90, primals_91, 786432, grid=grid(786432), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf164 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [out_54, batch_norm_29, out_55], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf164, buf160, primals_93, primals_94, 786432, grid=grid(786432), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf164, buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_48, relu_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf205, primals_151, primals_152, 786432, grid=grid(786432), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [out_92, batch_norm_49, out_93], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf207, buf203, primals_154, primals_155, 786432, grid=grid(786432), stream=stream0)
        del primals_154
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf43, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf248 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_68, relu_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf248, primals_212, primals_213, 786432, grid=grid(786432), stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [out_129], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf250 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [out_130, batch_norm_69, out_131], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf250, buf246, primals_215, primals_216, 786432, grid=grid(786432), stream=stream0)
        del primals_215
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf250, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_88, relu_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf291, primals_273, primals_274, 786432, grid=grid(786432), stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [out_167], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [out_168, batch_norm_89, out_169], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf293, buf289, primals_276, primals_277, 786432, grid=grid(786432), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf293, buf77, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_108, relu_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf334, primals_334, primals_335, 786432, grid=grid(786432), stream=stream0)
        del primals_334
        # Topologically Sorted Source Nodes: [out_205], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (4, 192, 32, 32), (196608, 1, 6144, 192))
        buf336 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [out_206, batch_norm_109, out_207], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf336, buf332, primals_337, primals_338, 786432, grid=grid(786432), stream=stream0)
        del primals_337
        # Topologically Sorted Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf336, buf94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_34, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_11, relu_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf124, primals_38, primals_39, 393216, grid=grid(393216), stream=stream0)
        del primals_38
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_95, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_31, relu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf167, primals_99, primals_100, 393216, grid=grid(393216), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_156, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_51, relu_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf210, primals_160, primals_161, 393216, grid=grid(393216), stream=stream0)
        del primals_160
        # Topologically Sorted Source Nodes: [out_95], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_217, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_71, relu_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf253, primals_221, primals_222, 393216, grid=grid(393216), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_278, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf296 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_91, relu_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf296, primals_282, primals_283, 393216, grid=grid(393216), stream=stream0)
        del primals_282
        # Topologically Sorted Source Nodes: [out_171], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 384, 16, 16), (98304, 1, 6144, 384))
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_339, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf339 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_111, relu_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf339, primals_343, primals_344, 393216, grid=grid(393216), stream=stream0)
        del primals_343
        # Topologically Sorted Source Nodes: [out_209], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_4, out_20, batch_norm_12, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf126, buf122, primals_35, primals_36, primals_41, primals_42, 393216, grid=grid(393216), stream=stream0)
        del buf122
        del primals_35
        del primals_41
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [input_10, out_58, batch_norm_32, out_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf169, buf165, primals_96, primals_97, primals_102, primals_103, 393216, grid=grid(393216), stream=stream0)
        del buf165
        del primals_102
        del primals_96
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [input_16, out_96, batch_norm_52, out_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf212, buf208, primals_157, primals_158, primals_163, primals_164, 393216, grid=grid(393216), stream=stream0)
        del buf208
        del primals_157
        del primals_163
        # Topologically Sorted Source Nodes: [out_98], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf255 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [input_22, out_134, batch_norm_72, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf255, buf251, primals_218, primals_219, primals_224, primals_225, 393216, grid=grid(393216), stream=stream0)
        del buf251
        del primals_218
        del primals_224
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf298 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [input_28, out_172, batch_norm_92, out_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf298, buf294, primals_279, primals_280, primals_285, primals_286, 393216, grid=grid(393216), stream=stream0)
        del buf294
        del primals_279
        del primals_285
        # Topologically Sorted Source Nodes: [out_174], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [input_34, out_210, batch_norm_112, out_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf341, buf337, primals_340, primals_341, primals_346, primals_347, 393216, grid=grid(393216), stream=stream0)
        del buf337
        del primals_340
        del primals_346
        # Topologically Sorted Source Nodes: [out_212], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_13, relu_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf128, primals_44, primals_45, 393216, grid=grid(393216), stream=stream0)
        del primals_44
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [out_24, batch_norm_14, out_25], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf130, buf126, primals_47, primals_48, 393216, grid=grid(393216), stream=stream0)
        del primals_47
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_33, relu_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf171, primals_105, primals_106, 393216, grid=grid(393216), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [out_62, batch_norm_34, out_63], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf173, buf169, primals_108, primals_109, 393216, grid=grid(393216), stream=stream0)
        del primals_108
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf173, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_53, relu_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf214, primals_166, primals_167, 393216, grid=grid(393216), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [out_99], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [out_100, batch_norm_54, out_101], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf216, buf212, primals_169, primals_170, 393216, grid=grid(393216), stream=stream0)
        del primals_169
        # Topologically Sorted Source Nodes: [out_102], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf216, buf47, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_73, relu_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf257, primals_227, primals_228, 393216, grid=grid(393216), stream=stream0)
        del primals_227
        # Topologically Sorted Source Nodes: [out_137], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf259 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [out_138, batch_norm_74, out_139], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf259, buf255, primals_230, primals_231, 393216, grid=grid(393216), stream=stream0)
        del primals_230
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf259, buf64, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_93, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf300, primals_288, primals_289, 393216, grid=grid(393216), stream=stream0)
        del primals_288
        # Topologically Sorted Source Nodes: [out_175], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf302 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [out_176, batch_norm_94, out_177], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf302, buf298, primals_291, primals_292, 393216, grid=grid(393216), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [out_178], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf302, buf81, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_113, relu_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf343, primals_349, primals_350, 393216, grid=grid(393216), stream=stream0)
        del primals_349
        # Topologically Sorted Source Nodes: [out_213], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 384, 16, 16), (98304, 1, 6144, 384))
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [out_214, batch_norm_114, out_215], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf345, buf341, primals_352, primals_353, 393216, grid=grid(393216), stream=stream0)
        del primals_352
        # Topologically Sorted Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf345, buf98, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_49, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_16, relu_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf133, primals_53, primals_54, 196608, grid=grid(196608), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_110, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_36, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf176, primals_114, primals_115, 196608, grid=grid(196608), stream=stream0)
        del primals_114
        # Topologically Sorted Source Nodes: [out_65], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_171, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_56, relu_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf219, primals_175, primals_176, 196608, grid=grid(196608), stream=stream0)
        del primals_175
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_232, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_76, relu_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf262, primals_236, primals_237, 196608, grid=grid(196608), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_293, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_96, relu_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf305, primals_297, primals_298, 196608, grid=grid(196608), stream=stream0)
        del primals_297
        # Topologically Sorted Source Nodes: [out_179], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 768, 8, 8), (49152, 1, 6144, 768))
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_354, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf348 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_116, relu_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf348, primals_358, primals_359, 196608, grid=grid(196608), stream=stream0)
        del primals_358
        # Topologically Sorted Source Nodes: [out_217], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_6, out_28, batch_norm_17, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf135, buf131, primals_50, primals_51, primals_56, primals_57, 196608, grid=grid(196608), stream=stream0)
        del buf131
        del primals_50
        del primals_56
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf178 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_12, out_66, batch_norm_37, out_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf178, buf174, primals_111, primals_112, primals_117, primals_118, 196608, grid=grid(196608), stream=stream0)
        del buf174
        del primals_111
        del primals_117
        # Topologically Sorted Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [input_18, out_104, batch_norm_57, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf221, buf217, primals_172, primals_173, primals_178, primals_179, 196608, grid=grid(196608), stream=stream0)
        del buf217
        del primals_172
        del primals_178
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [input_24, out_142, batch_norm_77, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf264, buf260, primals_233, primals_234, primals_239, primals_240, 196608, grid=grid(196608), stream=stream0)
        del buf260
        del primals_233
        del primals_239
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf307 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [input_30, out_180, batch_norm_97, out_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf307, buf303, primals_294, primals_295, primals_300, primals_301, 196608, grid=grid(196608), stream=stream0)
        del buf303
        del primals_294
        del primals_300
        # Topologically Sorted Source Nodes: [out_182], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [input_36, out_218, batch_norm_117, out_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf350, buf346, primals_355, primals_356, primals_361, primals_362, 196608, grid=grid(196608), stream=stream0)
        del buf346
        del primals_355
        del primals_361
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_18, relu_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf137, primals_59, primals_60, 196608, grid=grid(196608), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [out_32, batch_norm_19, out_33], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf139, buf135, primals_62, primals_63, 196608, grid=grid(196608), stream=stream0)
        del primals_62
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_38, relu_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf180, primals_120, primals_121, 196608, grid=grid(196608), stream=stream0)
        del primals_120
        # Topologically Sorted Source Nodes: [out_69], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [out_70, batch_norm_39, out_71], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf182, buf178, primals_123, primals_124, 196608, grid=grid(196608), stream=stream0)
        del primals_123
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_58, relu_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf223, primals_181, primals_182, 196608, grid=grid(196608), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [out_107], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [out_108, batch_norm_59, out_109], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf225, buf221, primals_184, primals_185, 196608, grid=grid(196608), stream=stream0)
        del primals_184
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_78, relu_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf266, primals_242, primals_243, 196608, grid=grid(196608), stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [out_145], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf268 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [out_146, batch_norm_79, out_147], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf268, buf264, primals_245, primals_246, 196608, grid=grid(196608), stream=stream0)
        del primals_245
        buf309 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_98, relu_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf309, primals_303, primals_304, 196608, grid=grid(196608), stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [out_183], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf311 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [out_184, batch_norm_99, out_185], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf311, buf307, primals_306, primals_307, 196608, grid=grid(196608), stream=stream0)
        del primals_306
        buf352 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_118, relu_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf352, primals_364, primals_365, 196608, grid=grid(196608), stream=stream0)
        del primals_364
        # Topologically Sorted Source Nodes: [out_221], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 768, 8, 8), (49152, 1, 6144, 768))
        buf354 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [out_222, batch_norm_119, out_223], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf354, buf350, primals_367, primals_368, 196608, grid=grid(196608), stream=stream0)
        del primals_367
        buf140 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf139, buf140, 12288, grid=grid(12288), stream=stream0)
        buf183 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_72], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf182, buf183, 12288, grid=grid(12288), stream=stream0)
        buf226 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf225, buf226, 12288, grid=grid(12288), stream=stream0)
        buf269 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf268, buf269, 12288, grid=grid(12288), stream=stream0)
        buf312 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_186], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf311, buf312, 12288, grid=grid(12288), stream=stream0)
        buf355 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_224], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf354, buf355, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 100, 2, 2), (400, 1, 200, 100))
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 100, 2, 2), (400, 1, 200, 100))
        # Topologically Sorted Source Nodes: [out_111], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 100, 2, 2), (400, 1, 200, 100))
        # Topologically Sorted Source Nodes: [out_149], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 100, 2, 2), (400, 1, 200, 100))
        # Topologically Sorted Source Nodes: [out_187], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 100, 2, 2), (400, 1, 200, 100))
        # Topologically Sorted Source Nodes: [out_225], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_369, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 100, 2, 2), (400, 1, 200, 100))
        buf142 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf141, buf142, 16, 100, grid=grid(16), stream=stream0)
        buf185 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf184, buf185, 16, 100, grid=grid(16), stream=stream0)
        buf228 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf227, buf228, 16, 100, grid=grid(16), stream=stream0)
        buf271 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf270, buf271, 16, 100, grid=grid(16), stream=stream0)
        buf314 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_189], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf313, buf314, 16, 100, grid=grid(16), stream=stream0)
        buf357 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_21.run(buf356, buf357, 16, 100, grid=grid(16), stream=stream0)
        buf143 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf142, buf143, 4, 4, grid=grid(4), stream=stream0)
        buf144 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf141, buf143, buf144, 16, 100, grid=grid(16), stream=stream0)
        buf186 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf185, buf186, 4, 4, grid=grid(4), stream=stream0)
        buf187 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf184, buf186, buf187, 16, 100, grid=grid(16), stream=stream0)
        buf229 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf228, buf229, 4, 4, grid=grid(4), stream=stream0)
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf227, buf229, buf230, 16, 100, grid=grid(16), stream=stream0)
        buf272 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf271, buf272, 4, 4, grid=grid(4), stream=stream0)
        buf273 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf270, buf272, buf273, 16, 100, grid=grid(16), stream=stream0)
        buf315 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_189], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf314, buf315, 4, 4, grid=grid(4), stream=stream0)
        buf316 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [out_189], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf313, buf315, buf316, 16, 100, grid=grid(16), stream=stream0)
        buf358 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_22.run(buf357, buf358, 4, 4, grid=grid(4), stream=stream0)
        buf359 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_23.run(buf356, buf358, buf359, 16, 100, grid=grid(16), stream=stream0)
        buf145 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf144, buf145, 4, 4, grid=grid(4), stream=stream0)
        del buf144
        buf367 = empty_strided_cuda((24, 400), (400, 1), torch.float32)
        buf361 = reinterpret_tensor(buf367, (4, 400), (400, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf141, buf143, buf145, buf361, 1600, grid=grid(1600), stream=stream0)
        buf188 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf187, buf188, 4, 4, grid=grid(4), stream=stream0)
        del buf187
        buf362 = reinterpret_tensor(buf367, (4, 400), (400, 1), 1600)  # alias
        # Topologically Sorted Source Nodes: [out_75], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf184, buf186, buf188, buf362, 1600, grid=grid(1600), stream=stream0)
        buf231 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf230, buf231, 4, 4, grid=grid(4), stream=stream0)
        del buf230
        buf363 = reinterpret_tensor(buf367, (4, 400), (400, 1), 3200)  # alias
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf227, buf229, buf231, buf363, 1600, grid=grid(1600), stream=stream0)
        buf274 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf273, buf274, 4, 4, grid=grid(4), stream=stream0)
        del buf273
        buf364 = reinterpret_tensor(buf367, (4, 400), (400, 1), 4800)  # alias
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf270, buf272, buf274, buf364, 1600, grid=grid(1600), stream=stream0)
        buf317 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_189], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf316, buf317, 4, 4, grid=grid(4), stream=stream0)
        del buf316
        buf365 = reinterpret_tensor(buf367, (4, 400), (400, 1), 6400)  # alias
        # Topologically Sorted Source Nodes: [out_189], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf313, buf315, buf317, buf365, 1600, grid=grid(1600), stream=stream0)
        buf360 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_24.run(buf359, buf360, 4, 4, grid=grid(4), stream=stream0)
        del buf359
        buf366 = reinterpret_tensor(buf367, (4, 400), (400, 1), 8000)  # alias
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_25.run(buf356, buf358, buf360, buf366, 1600, grid=grid(1600), stream=stream0)
        buf368 = empty_strided_cuda((4, 400), (400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_26.run(buf367, buf368, 1600, grid=grid(1600), stream=stream0)
        del buf361
        del buf362
        del buf363
        del buf364
        del buf365
        del buf366
        del buf367
    return (buf368, buf0, primals_6, buf1, primals_9, buf2, primals_12, buf3, primals_15, buf4, primals_18, primals_19, primals_21, buf5, primals_24, buf6, primals_27, buf7, primals_30, buf8, primals_33, primals_34, primals_36, buf9, primals_39, buf10, primals_42, buf11, primals_45, buf12, primals_48, primals_49, primals_51, buf13, primals_54, buf14, primals_57, buf15, primals_60, buf16, primals_63, primals_64, buf17, primals_67, buf18, primals_70, buf19, primals_73, buf20, primals_76, buf21, primals_79, primals_80, primals_82, buf22, primals_85, buf23, primals_88, buf24, primals_91, buf25, primals_94, primals_95, primals_97, buf26, primals_100, buf27, primals_103, buf28, primals_106, buf29, primals_109, primals_110, primals_112, buf30, primals_115, buf31, primals_118, buf32, primals_121, buf33, primals_124, primals_125, buf34, primals_128, buf35, primals_131, buf36, primals_134, buf37, primals_137, buf38, primals_140, primals_141, primals_143, buf39, primals_146, buf40, primals_149, buf41, primals_152, buf42, primals_155, primals_156, primals_158, buf43, primals_161, buf44, primals_164, buf45, primals_167, buf46, primals_170, primals_171, primals_173, buf47, primals_176, buf48, primals_179, buf49, primals_182, buf50, primals_185, primals_186, buf51, primals_189, buf52, primals_192, buf53, primals_195, buf54, primals_198, buf55, primals_201, primals_202, primals_204, buf56, primals_207, buf57, primals_210, buf58, primals_213, buf59, primals_216, primals_217, primals_219, buf60, primals_222, buf61, primals_225, buf62, primals_228, buf63, primals_231, primals_232, primals_234, buf64, primals_237, buf65, primals_240, buf66, primals_243, buf67, primals_246, primals_247, buf68, primals_250, buf69, primals_253, buf70, primals_256, buf71, primals_259, buf72, primals_262, primals_263, primals_265, buf73, primals_268, buf74, primals_271, buf75, primals_274, buf76, primals_277, primals_278, primals_280, buf77, primals_283, buf78, primals_286, buf79, primals_289, buf80, primals_292, primals_293, primals_295, buf81, primals_298, buf82, primals_301, buf83, primals_304, buf84, primals_307, primals_308, buf85, primals_311, buf86, primals_314, buf87, primals_317, buf88, primals_320, buf89, primals_323, primals_324, primals_326, buf90, primals_329, buf91, primals_332, buf92, primals_335, buf93, primals_338, primals_339, primals_341, buf94, primals_344, buf95, primals_347, buf96, primals_350, buf97, primals_353, primals_354, primals_356, buf98, primals_359, buf99, primals_362, buf100, primals_365, buf101, primals_368, primals_369, buf102, buf104, buf106, buf108, buf110, buf112, buf115, buf117, buf119, buf121, buf124, buf126, buf128, buf130, buf133, buf135, buf137, buf139, buf140, buf141, buf143, buf145, buf147, buf149, buf151, buf153, buf155, buf158, buf160, buf162, buf164, buf167, buf169, buf171, buf173, buf176, buf178, buf180, buf182, buf183, buf184, buf186, buf188, buf190, buf192, buf194, buf196, buf198, buf201, buf203, buf205, buf207, buf210, buf212, buf214, buf216, buf219, buf221, buf223, buf225, buf226, buf227, buf229, buf231, buf233, buf235, buf237, buf239, buf241, buf244, buf246, buf248, buf250, buf253, buf255, buf257, buf259, buf262, buf264, buf266, buf268, buf269, buf270, buf272, buf274, buf276, buf278, buf280, buf282, buf284, buf287, buf289, buf291, buf293, buf296, buf298, buf300, buf302, buf305, buf307, buf309, buf311, buf312, buf313, buf315, buf317, buf319, buf321, buf323, buf325, buf327, buf330, buf332, buf334, buf336, buf339, buf341, buf343, buf345, buf348, buf350, buf352, buf354, buf355, buf356, buf358, buf360, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((100, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
